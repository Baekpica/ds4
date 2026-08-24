/* Tool-schema / invoke helpers copied from ds4_server.c at v0.6.3-dfm.
 * Included by render_c_oracle.c. */

#include <stdint.h>

#define DS4_SOLAR_TOOL_START      "<|tool:start|>"
#define DS4_SOLAR_TOOL_END        "<|tool:end|>"
#define DS4_SOLAR_TOOL_CALL_START "<|tool_call:start|>"
#define DS4_SOLAR_TOOL_CALL_END   "<|tool_call:end|>"
#define DS4_SOLAR_TOOL_ARG_START  "<|tool_arg:start|>"
#define DS4_SOLAR_TOOL_ARG_VALUE  "<|tool_arg:value|>"
#define DS4_SOLAR_TOOL_ARG_END    "<|tool_arg:end|>"

typedef enum {
    TOOL_CHOICE_AUTO = 0,
    TOOL_CHOICE_NONE = 1,
    TOOL_CHOICE_REQUIRED = 2
} tool_choice_mode;

typedef struct {
    const char *name;
    const char **prop;
    int n_prop;
} oracle_order;

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("oom");
    return p;
}

static char *xstrdup(const char *s) {
    s = s ? s : "";
    char *p = strdup(s);
    if (!p) die("oom");
    return p;
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
        if (c != '\\') {
            buf_putc(&b, (char)c);
            continue;
        }
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
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff && json_u16(p, &lo) &&
                lo >= 0xdc00 && lo <= 0xdfff) {
                cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
            }
            utf8_put(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
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

#define JSON_MAX_NESTING 256
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

static char *json_minify_raw_value(const char *json) {
    const char *p = json ? json : "null";
    json_ws(&p);
    const char *start = p;
    if (!json_skip_value(&p)) return xstrdup(json ? json : "null");
    const char *end = p;
    buf b = {0};
    bool in_string = false;
    bool escape = false;
    for (const char *s = start; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (in_string) {
            buf_putc(&b, (char)c);
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
        } else if (c == '"') {
            in_string = true;
            buf_putc(&b, (char)c);
        } else if (!isspace(c)) {
            buf_putc(&b, (char)c);
        }
    }
    return buf_take(&b);
}

typedef struct {
    char *key;
    char *value;
    bool is_string;
    bool used;
} json_arg;

typedef struct {
    json_arg *v;
    int len;
    int cap;
} json_args;

static void json_args_free(json_args *args) {
    for (int i = 0; i < args->len; i++) {
        free(args->v[i].key);
        free(args->v[i].value);
    }
    free(args->v);
    memset(args, 0, sizeof(*args));
}

static void json_args_push(json_args *args, json_arg arg) {
    if (args->len == args->cap) {
        args->cap = args->cap ? args->cap * 2 : 8;
        json_arg *nv = realloc(args->v, (size_t)args->cap * sizeof(args->v[0]));
        if (!nv) die("oom");
        args->v = nv;
    }
    args->v[args->len++] = arg;
}

static int json_args_find_unused(json_args *args, const char *key) {
    if (!key) return -1;
    for (int i = 0; i < args->len; i++) {
        if (!args->v[i].used && args->v[i].key && !strcmp(args->v[i].key, key))
            return i;
    }
    return -1;
}

static bool json_args_parse(const char *json, json_args *args) {
    const char *p = json ? json : "";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        bool is_string = false;
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') goto bad;
        p++;
        json_ws(&p);
        if (*p == '"') {
            is_string = true;
            if (!json_string(&p, &value)) goto bad;
        } else {
            char *raw = NULL;
            if (!json_raw_value(&p, &raw)) goto bad;
            value = json_minify_raw_value(raw);
            free(raw);
        }
        json_arg arg = {.key = key, .value = value, .is_string = is_string};
        json_args_push(args, arg);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
        continue;
bad:
        free(key);
        free(value);
        json_args_free(args);
        return false;
    }
    if (*p != '}') {
        json_args_free(args);
        return false;
    }
    return true;
}

static void append_dsml_attr_escaped(buf *b, const char *s) {
    for (s = s ? s : ""; *s; s++) {
        if (*s == '&') buf_puts(b, "&amp;");
        else if (*s == '<') buf_puts(b, "&lt;");
        else if (*s == '>') buf_puts(b, "&gt;");
        else if (*s == '"') buf_puts(b, "&quot;");
        else buf_putc(b, *s);
    }
}

static void append_dsml_parameter_text(buf *b, const char *s) {
    const char *end = "</｜DSML｜parameter>";
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

static void append_dsml_json_literal(buf *b, const char *s) {
    const char *end = "</｜DSML｜parameter>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "\\u003c");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}

static void append_dsml_arg(buf *b, const json_arg *arg) {
    buf_puts(b, "<｜DSML｜parameter name=\"");
    append_dsml_attr_escaped(b, arg->key);
    buf_puts(b, "\" string=\"");
    buf_puts(b, arg->is_string ? "true" : "false");
    buf_puts(b, "\">");
    if (arg->is_string) append_dsml_parameter_text(b, arg->value);
    else append_dsml_json_literal(b, arg->value);
    buf_puts(b, "</｜DSML｜parameter>\n");
}

static bool append_dsml_arguments_from_json(buf *b, const char *json,
                                            const oracle_order *order) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) return false;
    if (order) {
        for (int i = 0; i < order->n_prop; i++) {
            int idx = json_args_find_unused(&args, order->prop[i]);
            if (idx < 0) continue;
            append_dsml_arg(b, &args.v[idx]);
            args.v[idx].used = true;
        }
    }
    for (int i = 0; i < args.len; i++) {
        if (args.v[i].used) continue;
        append_dsml_arg(b, &args.v[i]);
    }
    json_args_free(&args);
    return true;
}

static void append_dsml_tools_prompt_text(buf *b, const char *tool_schemas,
                                          bool tool_required) {
    if (!tool_schemas || !tool_schemas[0]) return;
    buf_puts(b,
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user question. "
        "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
        "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>\n\n"
        "String parameters should be specified as raw text and set `string=\"true\"`. "
        "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. "
        "Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. "
        "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n"
        "Otherwise, output directly after </think> with tool calls or final response.\n\n"
        "### Available Tool Schemas\n\n");
    buf_puts(b, tool_schemas);
    buf_puts(b, "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. "
                "Use the exact parameter names from the schemas.");
    if (tool_required) {
        buf_puts(b,
            "\n\n### Required Tool Use\n\n"
            "You MUST call at least one available tool in this turn. "
            "Do not end the turn with only reasoning or a final answer. "
            "After any reasoning, emit a valid <｜DSML｜tool_calls> block.");
    }
}

static void append_dsml_tool_calls_msg(buf *b, const chat_msg *m) {
    if (!m || m->n_calls <= 0) return;
    if (m->raw_dsml && m->raw_dsml[0]) {
        buf_puts(b, m->raw_dsml);
        return;
    }
    buf_puts(b, "\n\n<｜DSML｜tool_calls>\n");
    for (int i = 0; i < m->n_calls; i++) {
        const tool_call *tc = &m->calls[i];
        buf_puts(b, "<｜DSML｜invoke name=\"");
        append_dsml_attr_escaped(b, tc->name);
        buf_puts(b, "\">\n");
        if (!append_dsml_arguments_from_json(b, tc->arguments, NULL)) {
            buf_puts(b, "<｜DSML｜parameter name=\"arguments\" string=\"true\">");
            append_dsml_parameter_text(b, tc->arguments);
            buf_puts(b, "</｜DSML｜parameter>\n");
        }
        buf_puts(b, "</｜DSML｜invoke>\n");
    }
    buf_puts(b, "</｜DSML｜tool_calls>");
}

static void append_solar_tool_schema_blocks(buf *b, const char *tool_schemas) {
    const char *p = tool_schemas ? tool_schemas : "";
    while (*p) {
        json_ws(&p);
        if (!*p) break;
        char *schema = NULL;
        const char *before = p;
        if (!json_raw_value(&p, &schema)) {
            buf_puts(b, DS4_SOLAR_TOOL_START);
            buf_puts(b, before);
            buf_puts(b, DS4_SOLAR_TOOL_END "\n");
            return;
        }
        buf_puts(b, DS4_SOLAR_TOOL_START);
        buf_puts(b, schema);
        buf_puts(b, DS4_SOLAR_TOOL_END "\n");
        free(schema);
    }
}

static void append_solar_tools_prompt_text(buf *b, const char *tool_schemas) {
    if (!tool_schemas || !tool_schemas[0]) return;
    buf_puts(b,
        "## Tools\n"
        "- You may invoke one or more tools to assist with the user's query.\n\n"
        "### Available Tools\n");
    append_solar_tool_schema_blocks(b, tool_schemas);
    buf_puts(b,
        "\n### Tool Call Instruction\n"
        "- If using a tool, any reasoning must strictly precede the call. Do not append any text after the tool call.\n"
        "- If no tool is required, answer directly from your knowledge without ever mentioning the availability or absence of tools.\n"
        "- Each tool call MUST use this following format: "
        DS4_SOLAR_TOOL_CALL_START "{example-tool-name}\n"
        DS4_SOLAR_TOOL_ARG_START "{example-key-name-1}"
        DS4_SOLAR_TOOL_ARG_VALUE "{example-value-1}"
        DS4_SOLAR_TOOL_ARG_END "\n"
        DS4_SOLAR_TOOL_ARG_START "{example-key-name-2}"
        DS4_SOLAR_TOOL_ARG_VALUE "{example-value-2}"
        DS4_SOLAR_TOOL_ARG_END "\n"
        DS4_SOLAR_TOOL_CALL_END "\n");
}

static void append_solar_arg(buf *b, const json_arg *arg) {
    buf_puts(b, DS4_SOLAR_TOOL_ARG_START);
    buf_puts(b, arg->key ? arg->key : "");
    buf_puts(b, DS4_SOLAR_TOOL_ARG_VALUE);
    buf_puts(b, arg->value ? arg->value : "");
    buf_puts(b, DS4_SOLAR_TOOL_ARG_END "\n");
}

static bool append_solar_arguments_from_json(buf *b, const char *json,
                                             const oracle_order *order) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) return false;
    if (order) {
        for (int i = 0; i < order->n_prop; i++) {
            int idx = json_args_find_unused(&args, order->prop[i]);
            if (idx < 0) continue;
            append_solar_arg(b, &args.v[idx]);
            args.v[idx].used = true;
        }
    }
    for (int i = 0; i < args.len; i++) {
        if (!args.v[i].used) append_solar_arg(b, &args.v[i]);
    }
    json_args_free(&args);
    return true;
}

static const oracle_order *oracle_orders_find(const oracle_order *orders, int n,
                                              const char *name) {
    if (!orders || !name) return NULL;
    for (int i = 0; i < n; i++) {
        if (orders[i].name && !strcmp(orders[i].name, name)) return &orders[i];
    }
    return NULL;
}

static void append_solar_tool_calls_msg(buf *b, const chat_msg *m,
                                        const oracle_order *orders, int n_orders) {
    if (!m || m->n_calls <= 0) return;
    if (m->raw_dsml && m->raw_dsml[0]) {
        buf_puts(b, m->raw_dsml);
        return;
    }
    for (int i = 0; i < m->n_calls; i++) {
        const tool_call *tc = &m->calls[i];
        if (i) buf_putc(b, '\n');
        buf_puts(b, DS4_SOLAR_TOOL_CALL_START);
        buf_puts(b, tc->name ? tc->name : "");
        buf_putc(b, '\n');
        const oracle_order *order = oracle_orders_find(orders, n_orders, tc->name);
        if (!append_solar_arguments_from_json(b, tc->arguments, order)) {
            buf_puts(b, DS4_SOLAR_TOOL_ARG_START "arguments"
                        DS4_SOLAR_TOOL_ARG_VALUE);
            buf_puts(b, tc->arguments && tc->arguments[0] ? tc->arguments : "{}");
            buf_puts(b, DS4_SOLAR_TOOL_ARG_END "\n");
        }
        buf_puts(b, DS4_SOLAR_TOOL_CALL_END);
    }
}

static void append_motif3_tool_calls_msg(buf *out, const chat_msg *m, bool include_ids) {
    if (!m || m->n_calls <= 0) return;
    if (m->raw_tool_text && m->raw_tool_text[0]) {
        buf_puts(out, m->raw_tool_text);
        return;
    }
    for (int i = 0; i < m->n_calls; i++) {
        const tool_call *tc = &m->calls[i];
        buf_puts(out, "\n<tool_call>{\"name\": ");
        json_escape(out, tc->name ? tc->name : "");
        buf_puts(out, ", \"arguments\": ");
        if (tc->arguments && tc->arguments[0]) buf_puts(out, tc->arguments);
        else buf_puts(out, "null");
        if (include_ids && tc->id && tc->id[0]) {
            buf_puts(out, ", \"id\": ");
            json_escape(out, tc->id);
        }
        buf_puts(out, "}</tool_call>");
    }
}

static void append_exaone_tool_calls_msg(buf *out, const chat_msg *m, bool content_before) {
    if (!m || m->n_calls <= 0) return;
    if (m->raw_tool_text && m->raw_tool_text[0]) {
        const char *raw = m->raw_tool_text;
        while (*raw && isspace((unsigned char)*raw)) raw++;
        if (content_before) buf_putc(out, '\n');
        buf_puts(out, raw);
        return;
    }
    for (int i = 0; i < m->n_calls; i++) {
        const tool_call *tc = &m->calls[i];
        if (content_before || i) buf_putc(out, '\n');
        buf_puts(out, "<tool_call>{\"name\": ");
        json_escape(out, tc->name ? tc->name : "");
        buf_puts(out, ", \"arguments\": ");
        buf_puts(out, tc->arguments && tc->arguments[0] ? tc->arguments : "null");
        buf_puts(out, "}</tool_call>");
    }
}

static void append_exaone_tools_declaration(buf *out, const char *tool_schemas) {
    if (!tool_schemas || !tool_schemas[0]) return;
    buf_puts(out, "<|tool_declare|>\n# Tools\n");
    const char *p = tool_schemas;
    while (*p) {
        json_ws(&p);
        if (!*p) break;
        char *raw = NULL;
        if (!json_raw_value(&p, &raw)) {
            buf_puts(out, "<tool>");
            buf_puts(out, p);
            buf_puts(out, "</tool>\n");
            break;
        }
        char *schema = json_minify_raw_value(raw);
        buf_puts(out, "<tool>");
        buf_puts(out, schema ? schema : raw);
        buf_puts(out, "</tool>\n");
        free(schema);
        free(raw);
    }
    buf_puts(out, "<|endofturn|>\n");
}

static void append_motif3_tools_system_text(buf *out, const char *tool_schemas,
                                            const oracle_order *orders, int n_orders) {
    buf_puts(out,
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n\n"
        "<tools>");
    const char *p = tool_schemas ? tool_schemas : "";
    while (*p) {
        const char *end = strchr(p, '\n');
        buf_putc(out, '\n');
        if (end) {
            buf_append(out, p, (size_t)(end - p));
            p = end + 1;
        } else {
            buf_puts(out, p);
            break;
        }
    }
    buf_puts(out,
        "\n</tools>"
        "\n\nFor each function call, output in JSON within <tool_call> tags:\n");
    for (int i = 0; i < n_orders; i++) {
        const oracle_order *order = &orders[i];
        buf_puts(out, "\n<tool_call>{\"name\": ");
        json_escape(out, order->name ? order->name : "");
        buf_puts(out, ", \"arguments\": {");
        for (int j = 0; j < order->n_prop; j++) {
            if (j) buf_puts(out, ", ");
            json_escape(out, order->prop[j]);
            buf_puts(out, ": <");
            buf_puts(out, order->prop[j]);
            buf_putc(out, '>');
        }
        buf_puts(out, "}}</tool_call>");
    }
}

static void dots3_pyspace_json(buf *out, const char *raw) {
    const char *p = raw ? raw : "";
    bool in_string = false;
    while (*p) {
        const char c = *p;
        if (in_string) {
            buf_putc(out, c);
            if (c == '\\' && p[1]) {
                buf_putc(out, p[1]);
                p += 2;
                continue;
            }
            if (c == '"') in_string = false;
            p++;
            continue;
        }
        if (isspace((unsigned char)c)) {
            p++;
            continue;
        }
        buf_putc(out, c);
        if (c == '"') in_string = true;
        if (c == ',' || c == ':') buf_putc(out, ' ');
        p++;
    }
}

static void append_dots3_tool_call_text(buf *out, const tool_call *tc) {
    buf_puts(out, "\n<dots_function_call>\n<invoke name=\"");
    buf_puts(out, tc->name ? tc->name : "");
    buf_puts(out, "\">");
    const char *p = tc->arguments ? tc->arguments : "";
    json_ws(&p);
    if (*p == '{') {
        p++;
        json_ws(&p);
        while (*p && *p != '}') {
            char *key = NULL;
            if (!json_string(&p, &key)) break;
            json_ws(&p);
            if (*p != ':') { free(key); break; }
            p++;
            json_ws(&p);
            char *value = NULL;
            const bool is_string = *p == '"';
            char *plain = NULL;
            if (is_string) {
                if (!json_string(&p, &plain)) { free(key); break; }
            } else if (!json_raw_value(&p, &value)) {
                free(key);
                break;
            }
            buf_puts(out, "\n<parameter name=\"");
            buf_puts(out, key);
            buf_puts(out, "\">\n");
            if (is_string) buf_puts(out, plain ? plain : "");
            else dots3_pyspace_json(out, value);
            buf_puts(out, "\n</parameter>");
            free(plain);
            free(value);
            free(key);
            json_ws(&p);
            if (*p == ',') p++;
            json_ws(&p);
        }
    }
    buf_puts(out, "\n</invoke>\n</dots_function_call>");
}

static void append_dots3_tool_calls_msg(buf *out, const chat_msg *m) {
    if (!m) return;
    for (int i = 0; i < m->n_calls; i++)
        append_dots3_tool_call_text(out, &m->calls[i]);
}

static void append_dots3_tools_system_text(buf *out, const char *tool_schemas) {
    buf_puts(out,
        "\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> "
        "XML tags:\n<tools>");
    const char *p = tool_schemas ? tool_schemas : "";
    while (*p) {
        const char *end = strchr(p, '\n');
        const size_t n = end ? (size_t)(end - p) : strlen(p);
        char *line = xmalloc(n + 1);
        memcpy(line, p, n);
        line[n] = 0;
        buf_puts(out, "\n{\"type\": \"function\", \"function\": ");
        dots3_pyspace_json(out, line);
        buf_putc(out, '}');
        free(line);
        p += n;
        if (*p == '\n') p++;
    }
    buf_puts(out,
        "\n</tools>\n\n"
        "When making tool calls, use XML format to invoke tools and pass "
        "parameters:\n\n"
        "<dots_function_call>\n"
        "<invoke name=\"tool-name-1\">\n"
        "<parameter name=\"param-key-1\">\n"
        "param-value-1\n"
        "</parameter>\n"
        "<parameter name=\"param-key-2\">\n"
        "param-value-2\n"
        "</parameter>\n"
        "...\n"
        "</invoke>\n"
        "</dots_function_call>");
}

static char *render_dsml_tools(const chat_msgs *msgs, const char *tool_schemas,
                               ds4_think_mode think_mode, tool_choice_mode tool_choice) {
    const bool think = ds4_think_mode_enabled(think_mode);
    const bool tool_context = chat_history_uses_tool_context(msgs, tool_schemas);
    int last_user_idx = -1;
    buf system = {0};
    if (tool_schemas && tool_schemas[0]) {
        append_dsml_tools_prompt_text(&system, tool_schemas,
                                      tool_choice == TOOL_CHOICE_REQUIRED);
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!role_is_system(m->role)) continue;
        if (system.len) buf_puts(&system, "\n\n");
        buf_puts(&system, m->content ? m->content : "");
    }
    for (int i = 0; i < msgs->len; i++) {
        if (role_is_user_like(msgs->v[i].role)) last_user_idx = i;
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
            append_dsml_tool_calls_msg(&out, m);
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

static char *render_motif3_tools(const chat_msgs *msgs, const char *tool_schemas,
                                 const oracle_order *orders, int n_orders,
                                 ds4_think_mode think_mode) {
    const bool think = ds4_think_mode_enabled(think_mode);
    const bool have_tools = tool_schemas && tool_schemas[0];
    const bool first_system = msgs && msgs->len > 0 && role_is_system(msgs->v[0].role);
    int last_assistant = -1;
    for (int i = 0; msgs && i < msgs->len; i++)
        if (!strcmp(msgs->v[i].role, "assistant")) last_assistant = i;
    buf out = {0};
    buf_puts(&out, "<|beginoftext|>");
    if (have_tools) {
        buf_puts(&out, "<|startofturn|><|system|>");
        append_motif3_tools_system_text(&out, tool_schemas, orders, n_orders);
        if (first_system) {
            buf_puts(&out, "\n\n");
            buf_puts(&out, msgs->v[0].content ? msgs->v[0].content : "");
        }
        buf_puts(&out, "<|endofturn|>");
    } else if (first_system) {
        buf_puts(&out, "<|startofturn|><|system|>");
        buf_puts(&out, msgs->v[0].content ? msgs->v[0].content : "");
        buf_puts(&out, "<|endofturn|>");
    }
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (i == 0 && first_system) continue;
        else if (role_is_system(m->role)) {
            buf_puts(&out, "<|startofturn|><|system|>");
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, "<|endofturn|>");
        } else if (!strcmp(m->role, "user")) {
            buf_puts(&out, "<|startofturn|><|user|>");
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, "<|endofturn|>");
        } else if (!strcmp(m->role, "assistant")) {
            buf_puts(&out, "<|startofturn|><|assistant|>");
            if (m->reasoning && m->reasoning[0] &&
                (have_tools || i == last_assistant)) {
                buf_puts(&out, "<think>");
                motif3_buf_put_trimmed(&out, m->reasoning);
                buf_puts(&out, "</think>");
            }
            motif3_buf_put_trimmed(&out, m->content);
            append_motif3_tool_calls_msg(&out, m, true);
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

static char *render_dots3_tools(const chat_msgs *msgs, const char *tool_schemas,
                                ds4_think_mode think_mode) {
    const bool think = ds4_think_mode_enabled(think_mode);
    const bool have_tools = tool_schemas && tool_schemas[0];
    const bool first_system = msgs && msgs->len > 0 && role_is_system(msgs->v[0].role);
    buf out = {0};
    buf_puts(&out, "<|system|>");
    if (first_system) buf_puts(&out, msgs->v[0].content ? msgs->v[0].content : "");
    else buf_puts(&out, "You are a helpful assistant.");
    if (have_tools) append_dots3_tools_system_text(&out, tool_schemas);
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
            append_dots3_tool_calls_msg(&out, m);
            buf_puts(&out, "<|endofassistant|>");
            free(content_heap);
            free(reason_heap);
        } else if (m->role && (!strcmp(m->role, "tool") || !strcmp(m->role, "function"))) {
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

static char *render_exaone_tools(const chat_msgs *msgs, const char *tool_schemas,
                                 ds4_think_mode think_mode) {
    int last_user_idx = -1;
    for (int i = 0; msgs && i < msgs->len; i++) {
        if (!strcmp(msgs->v[i].role, "user") &&
            !chat_msg_is_model_tool_result(&msgs->v[i]))
            last_user_idx = i;
    }
    buf out = {0};
    append_exaone_tools_declaration(&out, tool_schemas);
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
            const bool have_content = exaone_buf_put_trimmed(&out, m->content);
            append_exaone_tool_calls_msg(&out, m, have_content);
            buf_puts(&out, "<|endofturn|>\n");
        }
    }
    buf_puts(&out, "<|assistant|>\n<think>\n");
    if (!ds4_think_mode_enabled(think_mode))
        buf_puts(&out, "\n</think>\n\n");
    return buf_take(&out);
}

static char *render_solar_tools(const chat_msgs *msgs, const char *tool_schemas,
                                const oracle_order *orders, int n_orders,
                                ds4_think_mode think_mode) {
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
    if (tool_schemas && tool_schemas[0]) {
        if (system.len) buf_puts(&system, "\n\n");
        append_solar_tools_prompt_text(&system, tool_schemas);
    }
    buf out = {0};
    if (system.len) {
        append_solar_role_open(&out, "system");
        buf_append(&out, system.ptr, system.len);
        buf_puts(&out, DS4_SOLAR_IM_END "\n");
    }
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) continue;
        else if (!strcmp(m->role, "user") && !chat_msg_is_model_tool_result(m)) {
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
            append_solar_tool_calls_msg(&out, m, orders, n_orders);
            if (m->n_calls > 0) buf_putc(&out, '\n');
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
