/* C KVC oracle for the Phase 4 4-way matrix.  Writes and reads the
 * envelope through ds4_kvstore.c — not a second format. */

#include "ds4_kvstore.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg) {
    fprintf(stderr, "kv_c_oracle: %s\n", msg);
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
    unsigned char *b = malloc(*len_out ? *len_out : 1);
    if (!b) die("oom");
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

static const char *need(int *i, int argc, char **argv, const char *flag) {
    if (*i + 1 >= argc) die(flag);
    return argv[++(*i)];
}

static int cmd_write(int argc, char **argv) {
    const char *path = NULL;
    const char *text_hex = NULL;
    const char *payload_hex = NULL;
    uint8_t model_id = 0, quant = 2, reason = 1, ext = 0;
    uint32_t tokens = 512, hits = 0, ctx = 2048;
    uint64_t created = 1, used = 1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--path")) path = need(&i, argc, argv, "--path");
        else if (!strcmp(argv[i], "--text-hex")) text_hex = need(&i, argc, argv, "--text-hex");
        else if (!strcmp(argv[i], "--payload-hex")) payload_hex = need(&i, argc, argv, "--payload-hex");
        else if (!strcmp(argv[i], "--model-id")) model_id = (uint8_t)atoi(need(&i, argc, argv, "--model-id"));
        else if (!strcmp(argv[i], "--quant")) quant = (uint8_t)atoi(need(&i, argc, argv, "--quant"));
        else if (!strcmp(argv[i], "--reason")) reason = (uint8_t)atoi(need(&i, argc, argv, "--reason"));
        else if (!strcmp(argv[i], "--ext")) ext = (uint8_t)atoi(need(&i, argc, argv, "--ext"));
        else if (!strcmp(argv[i], "--tokens")) tokens = (uint32_t)atoi(need(&i, argc, argv, "--tokens"));
        else if (!strcmp(argv[i], "--hits")) hits = (uint32_t)atoi(need(&i, argc, argv, "--hits"));
        else if (!strcmp(argv[i], "--ctx")) ctx = (uint32_t)atoi(need(&i, argc, argv, "--ctx"));
        else if (!strcmp(argv[i], "--created")) created = strtoull(need(&i, argc, argv, "--created"), NULL, 10);
        else if (!strcmp(argv[i], "--used")) used = strtoull(need(&i, argc, argv, "--used"), NULL, 10);
        else die("unknown write flag");
    }
    if (!path || !text_hex || !payload_hex) die("write needs --path --text-hex --payload-hex");
    size_t text_len = 0, payload_len = 0;
    unsigned char *text = parse_hex(text_hex, &text_len);
    unsigned char *payload = parse_hex(payload_hex, &payload_len);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, model_id, quant, reason, ext, tokens, hits, ctx,
                            created, used, (uint64_t)payload_len);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)text_len);
    FILE *fp = fopen(path, "wb");
    if (!fp) die("fopen write");
    int ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
             fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
             fwrite(text, 1, text_len, fp) == text_len &&
             fwrite(payload, 1, payload_len, fp) == payload_len;
    fclose(fp);
    free(text);
    free(payload);
    if (!ok) die("fwrite");
    return 0;
}

static int cmd_read(int argc, char **argv) {
    if (argc < 3) die("read PATH");
    const char *path = argv[2];
    FILE *fp = fopen(path, "rb");
    if (!fp) die("fopen read");
    ds4_kvstore_entry e = {0};
    uint32_t text_bytes = 0;
    if (!ds4_kvstore_read_header(fp, &e, &text_bytes)) {
        fclose(fp);
        die("read_header rejected the file");
    }
    unsigned char *text = malloc(text_bytes ? text_bytes : 1);
    unsigned char *payload = malloc(e.payload_bytes ? (size_t)e.payload_bytes : 1);
    if (!text || !payload) die("oom");
    if (fread(text, 1, text_bytes, fp) != text_bytes) die("short text");
    if (fread(payload, 1, (size_t)e.payload_bytes, fp) != (size_t)e.payload_bytes)
        die("short payload");
    fclose(fp);
    printf("model_id=%u\n", e.model_id);
    printf("quant_bits=%u\n", e.quant_bits);
    printf("reason=%u\n", e.reason);
    printf("ext_flags=%u\n", e.ext_flags);
    printf("tokens=%u\n", e.tokens);
    printf("hits=%u\n", e.hits);
    printf("ctx_size=%u\n", e.ctx_size);
    printf("created_at=%llu\n", (unsigned long long)e.created_at);
    printf("last_used=%llu\n", (unsigned long long)e.last_used);
    printf("payload_bytes=%llu\n", (unsigned long long)e.payload_bytes);
    printf("text_bytes=%u\n", text_bytes);
    printf("text_hex=");
    print_hex(text, text_bytes);
    printf("\npayload_hex=");
    print_hex(payload, (size_t)e.payload_bytes);
    printf("\n");
    free(text);
    free(payload);
    return 0;
}

static int cmd_sha1(int argc, char **argv) {
    if (argc < 3) die("sha1 --text-hex HEX");
    const char *hex = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--text-hex")) hex = need(&i, argc, argv, "--text-hex");
        else die("unknown sha1 flag");
    }
    if (!hex) die("sha1 needs --text-hex");
    size_t len = 0;
    unsigned char *b = parse_hex(hex, &len);
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(b, len, sha);
    printf("%s\n", sha);
    free(b);
    return 0;
}

static int cmd_score(int argc, char **argv) {
    ds4_kvstore_entry e = {0};
    uint64_t now = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--hits")) e.hits = (uint32_t)atoi(need(&i, argc, argv, "--hits"));
        else if (!strcmp(argv[i], "--tokens")) e.tokens = (uint32_t)atoi(need(&i, argc, argv, "--tokens"));
        else if (!strcmp(argv[i], "--file-size")) e.file_size = strtoull(need(&i, argc, argv, "--file-size"), NULL, 10);
        else if (!strcmp(argv[i], "--reason")) e.reason = (uint8_t)atoi(need(&i, argc, argv, "--reason"));
        else if (!strcmp(argv[i], "--created")) e.created_at = strtoull(need(&i, argc, argv, "--created"), NULL, 10);
        else if (!strcmp(argv[i], "--used")) e.last_used = strtoull(need(&i, argc, argv, "--used"), NULL, 10);
        else if (!strcmp(argv[i], "--now")) now = strtoull(need(&i, argc, argv, "--now"), NULL, 10);
        else die("unknown score flag");
    }
    double s = ds4_kvstore_entry_eviction_score(&e, NULL, now, NULL);
    printf("%.17g\n", s);
    return 0;
}

static int cmd_store_len(int argc, char **argv) {
    ds4_kvstore kc;
    memset(&kc, 0, sizeof(kc));
    kc.opt = ds4_kvstore_default_options();
    int tokens = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens")) tokens = atoi(need(&i, argc, argv, "--tokens"));
        else die("unknown store-len flag");
    }
    printf("%d\n", ds4_kvstore_store_len(&kc, tokens));
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) die("write|read|sha1|score|store-len");
    if (!strcmp(argv[1], "write")) return cmd_write(argc, argv);
    if (!strcmp(argv[1], "read")) return cmd_read(argc, argv);
    if (!strcmp(argv[1], "sha1")) return cmd_sha1(argc, argv);
    if (!strcmp(argv[1], "score")) return cmd_score(argc, argv);
    if (!strcmp(argv[1], "store-len")) return cmd_store_len(argc, argv);
    die("unknown command");
    return 2;
}
