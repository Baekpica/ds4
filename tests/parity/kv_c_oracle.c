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
    const char *trailer_hex = "";
    uint8_t model_id = 0, quant = 2, reason = 1, ext = 0;
    uint32_t tokens = 512, hits = 0, ctx = 2048;
    uint64_t created = 1, used = 1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--path")) path = need(&i, argc, argv, "--path");
        else if (!strcmp(argv[i], "--text-hex")) text_hex = need(&i, argc, argv, "--text-hex");
        else if (!strcmp(argv[i], "--payload-hex")) payload_hex = need(&i, argc, argv, "--payload-hex");
        else if (!strcmp(argv[i], "--trailer-hex")) trailer_hex = need(&i, argc, argv, "--trailer-hex");
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
    size_t text_len = 0, payload_len = 0, trailer_len = 0;
    unsigned char *text = parse_hex(text_hex, &text_len);
    unsigned char *payload = parse_hex(payload_hex, &payload_len);
    unsigned char *trailer = parse_hex(trailer_hex, &trailer_len);
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
             fwrite(payload, 1, payload_len, fp) == payload_len &&
             fwrite(trailer, 1, trailer_len, fp) == trailer_len;
    fclose(fp);
    free(text);
    free(payload);
    free(trailer);
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
    long payload_end = ftell(fp);
    if (payload_end < 0 || fseek(fp, 0, SEEK_END) != 0) die("measure trailer");
    long file_end = ftell(fp);
    if (file_end < payload_end) die("invalid trailer length");
    size_t trailer_len = (size_t)(file_end - payload_end);
    unsigned char *trailer = malloc(trailer_len ? trailer_len : 1);
    if (!trailer) die("oom");
    if (fseek(fp, payload_end, SEEK_SET) != 0) die("seek trailer");
    if (fread(trailer, 1, trailer_len, fp) != trailer_len) die("short trailer");
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
    printf("\ntrailer_hex=");
    print_hex(trailer, trailer_len);
    printf("\n");
    free(text);
    free(payload);
    free(trailer);
    return 0;
}

static int cmd_ktm_encode(int argc, char **argv) {
    if ((argc - 2) % 2 != 0) die("ktm-encode needs ID_HEX DSML_HEX pairs");
    if (argc == 2) {
        printf("\n");
        return 0;
    }
    uint8_t header[8] = {'K', 'T', 'M', 1, 0, 0, 0, 0};
    ds4_kvstore_le_put32(header + 4, (uint32_t)((argc - 2) / 2));
    print_hex(header, sizeof(header));
    for (int i = 2; i < argc; i += 2) {
        size_t id_len = 0, dsml_len = 0;
        unsigned char *id = parse_hex(argv[i], &id_len);
        unsigned char *dsml = parse_hex(argv[i + 1], &dsml_len);
        if (id_len == 0 || dsml_len == 0 || id_len > UINT32_MAX || dsml_len > UINT32_MAX)
            die("invalid ktm entry");
        uint8_t lens[8];
        ds4_kvstore_le_put32(lens, (uint32_t)id_len);
        ds4_kvstore_le_put32(lens + 4, (uint32_t)dsml_len);
        print_hex(lens, sizeof(lens));
        print_hex(id, id_len);
        print_hex(dsml, dsml_len);
        free(id);
        free(dsml);
    }
    printf("\n");
    return 0;
}

static int cmd_ktm_decode(int argc, char **argv) {
    if (argc < 3 || argc > 4) die("ktm-decode TRAILER_HEX [MAX_IDS]");
    size_t len = 0;
    unsigned char *data = parse_hex(argv[2], &len);
    uint64_t max_ids = argc == 4 ? strtoull(argv[3], NULL, 10) : 100000;
    if (max_ids == 0) max_ids = 100000;
    int loaded = 0;
    size_t pos = 8;
    if (len < 8 || memcmp(data, "KTM\x01", 4) != 0) goto done;
    uint32_t count = ds4_kvstore_le_get32(data + 4);
    if ((uint64_t)count > max_ids * 4u) goto done;
    for (uint32_t i = 0; i < count; i++) {
        if (len - pos < 8) break;
        uint32_t id_len = ds4_kvstore_le_get32(data + pos);
        uint32_t dsml_len = ds4_kvstore_le_get32(data + pos + 4);
        pos += 8;
        if (id_len == 0 || id_len > 256 || dsml_len == 0 ||
            dsml_len > 512u * 1024u * 1024u ||
            (uint64_t)id_len + dsml_len > len - pos)
            break;
        unsigned char *id_zero = memchr(data + pos, 0, id_len);
        size_t id_text_len = id_zero ? (size_t)(id_zero - (data + pos)) : id_len;
        unsigned char *dsml = data + pos + id_len;
        unsigned char *dsml_zero = memchr(dsml, 0, dsml_len);
        size_t dsml_text_len = dsml_zero ? (size_t)(dsml_zero - dsml) : dsml_len;
        print_hex(data + pos, id_text_len);
        printf(":");
        print_hex(dsml, dsml_text_len);
        printf("\n");
        pos += (size_t)id_len + dsml_len;
        loaded++;
    }
done:
    printf("loaded=%d\n", loaded);
    free(data);
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

static int cmd_continued_target(int argc, char **argv) {
    ds4_kvstore kc;
    memset(&kc, 0, sizeof(kc));
    kc.enabled = true;
    kc.opt = ds4_kvstore_default_options();
    int live_tokens = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--min")) kc.opt.min_tokens = atoi(need(&i, argc, argv, "--min"));
        else if (!strcmp(argv[i], "--interval")) kc.opt.continued_interval_tokens = atoi(need(&i, argc, argv, "--interval"));
        else if (!strcmp(argv[i], "--align")) kc.opt.boundary_align_tokens = atoi(need(&i, argc, argv, "--align"));
        else if (!strcmp(argv[i], "--last")) kc.continued_last_store_tokens = atoi(need(&i, argc, argv, "--last"));
        else if (!strcmp(argv[i], "--live")) live_tokens = atoi(need(&i, argc, argv, "--live"));
        else die("unknown continued-target flag");
    }
    printf("%d\n", ds4_kvstore_continued_store_target(&kc, live_tokens));
    return 0;
}

static int cmd_chat_anchor(int argc, char **argv) {
    if (argc < 4) die("chat-anchor USER ASSISTANT [TOKENS...]");
    ds4_kvstore kc;
    memset(&kc, 0, sizeof(kc));
    kc.opt = ds4_kvstore_default_options();
    ds4_tokens prompt = {0};
    prompt.len = prompt.cap = argc - 4;
    prompt.v = calloc((size_t)(prompt.len ? prompt.len : 1), sizeof(*prompt.v));
    if (!prompt.v) die("oom");
    for (int i = 0; i < prompt.len; i++) prompt.v[i] = atoi(argv[i + 4]);
    printf("%d\n", ds4_kvstore_chat_anchor_pos(&kc, &prompt,
                                                atoi(argv[2]), atoi(argv[3])));
    free(prompt.v);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) die("write|read|ktm-encode|ktm-decode|sha1|score|store-len|continued-target|chat-anchor");
    if (!strcmp(argv[1], "write")) return cmd_write(argc, argv);
    if (!strcmp(argv[1], "read")) return cmd_read(argc, argv);
    if (!strcmp(argv[1], "ktm-encode")) return cmd_ktm_encode(argc, argv);
    if (!strcmp(argv[1], "ktm-decode")) return cmd_ktm_decode(argc, argv);
    if (!strcmp(argv[1], "sha1")) return cmd_sha1(argc, argv);
    if (!strcmp(argv[1], "score")) return cmd_score(argc, argv);
    if (!strcmp(argv[1], "store-len")) return cmd_store_len(argc, argv);
    if (!strcmp(argv[1], "continued-target")) return cmd_continued_target(argc, argv);
    if (!strcmp(argv[1], "chat-anchor")) return cmd_chat_anchor(argc, argv);
    die("unknown command");
    return 2;
}
