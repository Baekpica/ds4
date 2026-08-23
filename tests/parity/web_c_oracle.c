/* C encode/wire oracle for Phase 5. Helpers are copied from ds4_web.c
 * at v0.6.3-dfm so the Rust crate can compare bytes without linking
 * the Chrome subprocess path. */

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} web_buf;

static void *web_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { perror("malloc"); exit(1); }
    return p;
}

static char *web_xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = web_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void web_buf_append(web_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        char *p = realloc(b->ptr, cap);
        if (!p) { perror("realloc"); exit(1); }
        b->ptr = p;
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void web_buf_puts(web_buf *b, const char *s) {
    web_buf_append(b, s, strlen(s));
}

static char *web_buf_take(web_buf *b) {
    if (!b->ptr) return web_xstrdup("");
    char *p = b->ptr;
    b->ptr = NULL;
    b->len = b->cap = 0;
    return p;
}

static char *web_url_encode(const char *s) {
    static const char hex[] = "0123456789ABCDEF";
    web_buf b = {0};
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        unsigned char c = *p;
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            web_buf_append(&b, (const char *)&c, 1);
        } else {
            char e[3] = {'%', hex[c >> 4], hex[c & 15]};
            web_buf_append(&b, e, 3);
        }
    }
    return web_buf_take(&b);
}

static char *web_base64(const unsigned char *data, size_t len) {
    static const char tab[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t outlen = ((len + 2) / 3) * 4;
    char *out = web_xmalloc(outlen + 1);
    size_t j = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) v |= data[i + 2];
        out[j++] = tab[(v >> 18) & 63];
        out[j++] = tab[(v >> 12) & 63];
        out[j++] = (i + 1 < len) ? tab[(v >> 6) & 63] : '=';
        out[j++] = (i + 2 < len) ? tab[v & 63] : '=';
    }
    out[j] = '\0';
    return out;
}

static char *web_json_quote(const char *s) {
    web_buf b = {0};
    web_buf_puts(&b, "\"");
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        unsigned char c = *p;
        switch (c) {
        case '\\': web_buf_puts(&b, "\\\\"); break;
        case '"': web_buf_puts(&b, "\\\""); break;
        case '\n': web_buf_puts(&b, "\\n"); break;
        case '\r': web_buf_puts(&b, "\\r"); break;
        case '\t': web_buf_puts(&b, "\\t"); break;
        default:
            if (c < 0x20) {
                char tmp[8];
                snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                web_buf_puts(&b, tmp);
            } else {
                web_buf_append(&b, (const char *)&c, 1);
            }
            break;
        }
    }
    web_buf_puts(&b, "\"");
    return web_buf_take(&b);
}

static int web_hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        int x;
        if (c >= '0' && c <= '9') x = c - '0';
        else if (c >= 'a' && c <= 'f') x = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') x = c - 'A' + 10;
        else return -1;
        v = (v << 4) | x;
    }
    return v;
}

static void web_utf8_append(web_buf *b, unsigned code) {
    char out[4];
    if (code <= 0x7f) {
        out[0] = (char)code;
        web_buf_append(b, out, 1);
    } else if (code <= 0x7ff) {
        out[0] = (char)(0xc0 | (code >> 6));
        out[1] = (char)(0x80 | (code & 0x3f));
        web_buf_append(b, out, 2);
    } else if (code <= 0xffff) {
        out[0] = (char)(0xe0 | (code >> 12));
        out[1] = (char)(0x80 | ((code >> 6) & 0x3f));
        out[2] = (char)(0x80 | (code & 0x3f));
        web_buf_append(b, out, 3);
    } else {
        out[0] = (char)(0xf0 | (code >> 18));
        out[1] = (char)(0x80 | ((code >> 12) & 0x3f));
        out[2] = (char)(0x80 | ((code >> 6) & 0x3f));
        out[3] = (char)(0x80 | (code & 0x3f));
        web_buf_append(b, out, 4);
    }
}

static char *web_json_parse_string_at(const char *q, const char **endp) {
    if (*q != '"') return NULL;
    q++;
    web_buf b = {0};
    while (*q && *q != '"') {
        if (*q != '\\') {
            web_buf_append(&b, q++, 1);
            continue;
        }
        q++;
        switch (*q) {
        case '"': web_buf_append(&b, "\"", 1); q++; break;
        case '\\': web_buf_append(&b, "\\", 1); q++; break;
        case '/': web_buf_append(&b, "/", 1); q++; break;
        case 'b': web_buf_append(&b, "\b", 1); q++; break;
        case 'f': web_buf_append(&b, "\f", 1); q++; break;
        case 'n': web_buf_append(&b, "\n", 1); q++; break;
        case 'r': web_buf_append(&b, "\r", 1); q++; break;
        case 't': web_buf_append(&b, "\t", 1); q++; break;
        case 'u': {
            int v = web_hex4(q + 1);
            if (v < 0) { free(b.ptr); return NULL; }
            q += 5;
            if (v >= 0xd800 && v <= 0xdbff && q[0] == '\\' && q[1] == 'u') {
                int lo = web_hex4(q + 2);
                if (lo >= 0xdc00 && lo <= 0xdfff) {
                    unsigned code = 0x10000 + (((unsigned)v - 0xd800) << 10) +
                                    ((unsigned)lo - 0xdc00);
                    web_utf8_append(&b, code);
                    q += 6;
                    break;
                }
            }
            web_utf8_append(&b, (unsigned)v);
            break;
        }
        default:
            if (*q) web_buf_append(&b, q++, 1);
            break;
        }
    }
    if (*q != '"') {
        free(b.ptr);
        return NULL;
    }
    if (endp) *endp = q + 1;
    return web_buf_take(&b);
}

static char *web_json_get_string(const char *json, const char *key) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = json;
    while ((p = strstr(p, pat)) != NULL) {
        p += strlen(pat);
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p++ != ':') continue;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '"') return web_json_parse_string_at(p, NULL);
    }
    return NULL;
}

static int web_json_id_matches(const char *json, int id) {
    const char *p = strstr(json, "\"id\"");
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p) == id;
}

static void die(const char *msg) {
    fprintf(stderr, "web_c_oracle: %s\n", msg);
    exit(2);
}

static int hex_nibble(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static unsigned char *parse_hex(const char *s, size_t *len_out) {
    size_t n = strlen(s);
    if (n % 2 != 0) die("hex length is odd");
    *len_out = n / 2;
    unsigned char *b = web_xmalloc(*len_out ? *len_out : 1);
    for (size_t i = 0; i < *len_out; i++) {
        int hi = hex_nibble((unsigned char)s[i * 2]);
        int lo = hex_nibble((unsigned char)s[i * 2 + 1]);
        if (hi < 0 || lo < 0) die("invalid hex");
        b[i] = (unsigned char)((hi << 4) | lo);
    }
    return b;
}

static void print_hex(const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%02x", p[i]);
}

static void ws_text_frame(const unsigned char *text, size_t len, const unsigned char mask[4]) {
    unsigned char hdr[14];
    size_t h = 0;
    hdr[h++] = 0x81;
    if (len < 126) {
        hdr[h++] = 0x80 | (unsigned char)len;
    } else if (len <= 0xffff) {
        hdr[h++] = 0x80 | 126;
        hdr[h++] = (unsigned char)(len >> 8);
        hdr[h++] = (unsigned char)len;
    } else {
        hdr[h++] = 0x80 | 127;
        for (int i = 7; i >= 0; i--) hdr[h++] = (unsigned char)((uint64_t)len >> (i * 8));
    }
    for (int i = 0; i < 4; i++) hdr[h++] = mask[i];
    print_hex(hdr, h);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = text[i] ^ mask[i & 3];
        printf("%02x", c);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: web_c_oracle <cmd> ...");
    const char *cmd = argv[1];
    if (!strcmp(cmd, "url-encode")) {
        if (argc < 3) die("url-encode TEXT");
        char *s = web_url_encode(argv[2]);
        fputs(s, stdout);
        free(s);
    } else if (!strcmp(cmd, "base64")) {
        if (argc < 3) die("base64 HEX");
        size_t n = 0;
        unsigned char *b = parse_hex(argv[2], &n);
        char *s = web_base64(b, n);
        fputs(s, stdout);
        free(s);
        free(b);
    } else if (!strcmp(cmd, "json-quote")) {
        if (argc < 3) die("json-quote TEXT");
        char *s = web_json_quote(argv[2]);
        fputs(s, stdout);
        free(s);
    } else if (!strcmp(cmd, "json-get")) {
        if (argc < 4) die("json-get JSON KEY");
        char *s = web_json_get_string(argv[2], argv[3]);
        if (!s) die("missing key");
        fwrite(s, 1, strlen(s), stdout);
        free(s);
    } else if (!strcmp(cmd, "json-id")) {
        if (argc < 4) die("json-id JSON ID");
        printf("%s", web_json_id_matches(argv[2], atoi(argv[3])) ? "yes" : "no");
    } else if (!strcmp(cmd, "http-req")) {
        if (argc < 5) die("http-req METHOD PORT PATH");
        char line[512];
        snprintf(line, sizeof(line),
                 "%s %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nConnection: close\r\n\r\n",
                 argv[2], argv[4], atoi(argv[3]));
        fputs(line, stdout);
    } else if (!strcmp(cmd, "ws-handshake")) {
        if (argc < 6) die("ws-handshake PATH HOST PORT KEY");
        char line[512];
        snprintf(line, sizeof(line),
                 "GET %s HTTP/1.1\r\n"
                 "Host: %s:%d\r\n"
                 "Upgrade: websocket\r\n"
                 "Connection: Upgrade\r\n"
                 "Sec-WebSocket-Key: %s\r\n"
                 "Sec-WebSocket-Version: 13\r\n\r\n",
                 argv[2], argv[3], atoi(argv[4]), argv[5]);
        fputs(line, stdout);
    } else if (!strcmp(cmd, "cdp-req")) {
        if (argc < 4) die("cdp-req ID METHOD [PARAMS]");
        web_buf req = {0};
        char head[256];
        snprintf(head, sizeof(head), "{\"id\":%d,\"method\":", atoi(argv[2]));
        web_buf_puts(&req, head);
        char *qmethod = web_json_quote(argv[3]);
        web_buf_puts(&req, qmethod);
        free(qmethod);
        if (argc >= 5 && argv[4][0]) {
            web_buf_puts(&req, ",\"params\":");
            web_buf_puts(&req, argv[4]);
        }
        web_buf_puts(&req, "}");
        char *wire = web_buf_take(&req);
        fputs(wire, stdout);
        free(wire);
    } else if (!strcmp(cmd, "eval-params")) {
        if (argc < 3) die("eval-params EXPR");
        char *qexpr = web_json_quote(argv[2]);
        web_buf params = {0};
        web_buf_puts(&params, "{\"expression\":");
        web_buf_puts(&params, qexpr);
        web_buf_puts(&params, ",\"returnByValue\":true,\"awaitPromise\":true,\"includeCommandLineAPI\":true}");
        free(qexpr);
        char *s = web_buf_take(&params);
        fputs(s, stdout);
        free(s);
    } else if (!strcmp(cmd, "navigate-params")) {
        if (argc < 3) die("navigate-params URL");
        char *qurl = web_json_quote(argv[2]);
        web_buf params = {0};
        web_buf_puts(&params, "{\"url\":");
        web_buf_puts(&params, qurl);
        web_buf_puts(&params, "}");
        free(qurl);
        char *s = web_buf_take(&params);
        fputs(s, stdout);
        free(s);
    } else if (!strcmp(cmd, "create-target-params")) {
        if (argc < 3) die("create-target-params URL");
        char *qurl = web_json_quote(argv[2]);
        web_buf params = {0};
        web_buf_puts(&params, "{\"url\":");
        web_buf_puts(&params, qurl);
        web_buf_puts(&params, ",\"background\":true,\"newWindow\":false}");
        free(qurl);
        char *s = web_buf_take(&params);
        fputs(s, stdout);
        free(s);
    } else if (!strcmp(cmd, "search-url")) {
        if (argc < 3) die("search-url QUERY");
        char *q = web_url_encode(argv[2]);
        web_buf url = {0};
        web_buf_puts(&url, "https://www.google.com/search?q=");
        web_buf_puts(&url, q);
        free(q);
        char *s = web_buf_take(&url);
        fputs(s, stdout);
        free(s);
    } else if (!strcmp(cmd, "ws-frame")) {
        if (argc < 4) die("ws-frame TEXT-HEX MASK-HEX");
        size_t n = 0, mn = 0;
        unsigned char *text = parse_hex(argv[2], &n);
        unsigned char *maskb = parse_hex(argv[3], &mn);
        if (mn != 4) die("mask must be 4 bytes");
        ws_text_frame(text, n, maskb);
        free(text);
        free(maskb);
    } else if (!strcmp(cmd, "chrome-linux-args")) {
        if (argc < 5) die("chrome-linux-args PORT PROFILE root|user");
        int port = atoi(argv[2]);
        const char *profile = argv[3];
        int root = !strcmp(argv[4], "root");
        printf("--remote-debugging-port=%d\n", port);
        printf("--remote-allow-origins=*\n");
        printf("--user-data-dir=%s\n", profile);
        printf("--no-first-run\n");
        printf("--no-default-browser-check\n");
        printf("--disable-sync\n");
        printf("--password-store=basic\n");
        if (root) printf("--no-sandbox\n");
        printf("--mute-audio\n");
        printf("about:blank\n");
    } else {
        die("unknown command");
    }
    return 0;
}
