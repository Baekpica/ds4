/* DSV4 host-prefix oracle. Header + token ids from ds4.c at v0.6.3-dfm.
 * GPU / logits tails are not generated here. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAGIC UINT32_C(0x34565344)
#define VERSION UINT32_C(3)
#define NFIELD 13u
#define LAYOUT_SOLAR UINT32_C(0x33524c53)
#define LAYOUT_EXAONE UINT32_C(0x33415845)
#define LAYOUT_MOTIF3 UINT32_C(0x3346544d)
#define LAYOUT_DOTS3 UINT32_C(0x33535444)

static void put_u32(uint8_t out[4], uint32_t v)
{
    out[0] = (uint8_t)v;
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t get_u32(const uint8_t in[4])
{
    return (uint32_t)in[0] | ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) | ((uint32_t)in[3] << 24);
}

static void hex(const uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) printf("%02x", p[i]);
    printf("\n");
}

static size_t encode(uint8_t *out, const uint32_t h[NFIELD],
                     const uint32_t *tok, uint32_t ntok)
{
    for (uint32_t i = 0; i < NFIELD; i++) put_u32(out + i * 4u, h[i]);
    for (uint32_t i = 0; i < ntok; i++) put_u32(out + (NFIELD + i) * 4u, tok[i]);
    return (size_t)(NFIELD + ntok) * 4u;
}

static const char *layout_name(uint32_t f5)
{
    if (f5 == LAYOUT_SOLAR) return "solar-open2";
    if (f5 == LAYOUT_EXAONE) return "exaone-moe";
    if (f5 == LAYOUT_MOTIF3) return "motif3";
    if (f5 == LAYOUT_DOTS3) return "dots3-note";
    return "deepseek4";
}

static int parse(const uint8_t *b, size_t n, uint32_t h[NFIELD],
                 uint32_t *tok, uint32_t *ntok, const char **err)
{
    if (n < (size_t)NFIELD * 4u) {
        *err = "truncated session payload";
        return 1;
    }
    for (uint32_t i = 0; i < NFIELD; i++) h[i] = get_u32(b + i * 4u);
    if (h[0] != MAGIC || h[1] == 0 || h[1] > VERSION) {
        *err = "unsupported session payload version";
        return 1;
    }
    uint32_t nt = h[7];
    size_t need = (size_t)NFIELD * 4u + (size_t)nt * 4u;
    if (n < need) {
        *err = "truncated session payload";
        return 1;
    }
    if (h[5] == LAYOUT_SOLAR || h[5] == LAYOUT_EXAONE ||
        h[5] == LAYOUT_MOTIF3 || h[5] == LAYOUT_DOTS3) {
        if (h[12] != h[7]) {
            *err = "session payload token count does not match live rows";
            return 1;
        }
    }
    for (uint32_t i = 0; i < nt; i++) tok[i] = get_u32(b + (NFIELD + i) * 4u);
    *ntok = nt;
    return 0;
}

static void inspect(const uint32_t h[NFIELD], const uint32_t *tok, uint32_t ntok)
{
    size_t prefix = (size_t)(NFIELD + ntok) * 4u;
    printf("layout=%s version=%u ctx=%u ntok=%u prefix=%zu\n",
           layout_name(h[5]), h[1], h[2], ntok, prefix);
    printf("tokens=");
    for (uint32_t i = 0; i < ntok; i++) {
        if (i) printf(",");
        printf("%u", tok[i]);
    }
    printf("\n");
}

static void fill_deepseek(uint32_t h[NFIELD], uint32_t tok[3])
{
    memset(h, 0, sizeof(uint32_t) * NFIELD);
    h[0] = MAGIC; h[1] = VERSION; h[2] = 8192; h[3] = 64;
    h[4] = 8192; h[5] = 512; h[6] = 2048; h[7] = 3;
    h[8] = 61; h[9] = 128; h[10] = 64; h[11] = 129280; h[12] = 3;
    tok[0] = 10; tok[1] = 20; tok[2] = 30;
}

static void fill_cpu(uint32_t h[NFIELD], uint32_t tok[2])
{
    memset(h, 0, sizeof(uint32_t) * NFIELD);
    h[0] = MAGIC; h[1] = VERSION; h[2] = 8192; h[3] = 64;
    h[4] = 8192; h[5] = 8192; h[6] = 2048; h[7] = 2;
    h[8] = 61; h[9] = 128; h[10] = 64; h[11] = 129280; h[12] = 2;
    tok[0] = 1; tok[1] = 2;
}

static void fill_solar(uint32_t h[NFIELD], uint32_t tok[2])
{
    memset(h, 0, sizeof(uint32_t) * NFIELD);
    h[0] = MAGIC; h[1] = VERSION; h[2] = 4096; h[3] = 1;
    h[4] = 4096; h[5] = LAYOUT_SOLAR; h[6] = 1000; h[7] = 2;
    h[8] = 48; h[9] = 256; h[10] = 12; h[11] = 200064; h[12] = 2;
    tok[0] = 7; tok[1] = 8;
}

static void fill_exaone(uint32_t h[NFIELD], uint32_t tok[2])
{
    memset(h, 0, sizeof(uint32_t) * NFIELD);
    h[0] = MAGIC; h[1] = VERSION; h[2] = 8192; h[3] = 64;
    h[4] = 48; h[5] = LAYOUT_EXAONE; h[6] = 512; h[7] = 2;
    h[8] = 49; h[9] = 128; h[10] = 128; h[11] = 153088; h[12] = 2;
    tok[0] = 1; tok[1] = 2;
}

static void fill_motif3(uint32_t h[NFIELD], uint32_t tok[2])
{
    memset(h, 0, sizeof(uint32_t) * NFIELD);
    h[0] = MAGIC; h[1] = VERSION; h[2] = 8192; h[3] = 2048;
    h[4] = 512; h[5] = LAYOUT_MOTIF3; h[6] = 1024; h[7] = 2;
    h[8] = 53; h[9] = 64; h[10] = 128; h[11] = 151936; h[12] = 2;
    tok[0] = 3; tok[1] = 4;
}

static void fill_dots3(uint32_t h[NFIELD], uint32_t tok[2])
{
    memset(h, 0, sizeof(uint32_t) * NFIELD);
    h[0] = MAGIC; h[1] = VERSION; h[2] = 8192; h[3] = 2048;
    h[4] = 512; h[5] = LAYOUT_DOTS3; h[6] = 1024; h[7] = 2;
    h[8] = 46; h[9] = 64; h[10] = 128; h[11] = 152064; h[12] = 2;
    tok[0] = 5; tok[1] = 6;
}

static void script_encode_deepseek(void)
{
    uint32_t h[NFIELD], tok[3];
    uint8_t buf[128];
    fill_deepseek(h, tok);
    hex(buf, encode(buf, h, tok, 3));
}

static void script_encode_cpu(void)
{
    uint32_t h[NFIELD], tok[2];
    uint8_t buf[128];
    fill_cpu(h, tok);
    hex(buf, encode(buf, h, tok, 2));
}

static void script_encode_solar(void)
{
    uint32_t h[NFIELD], tok[2];
    uint8_t buf[128];
    fill_solar(h, tok);
    hex(buf, encode(buf, h, tok, 2));
}

static void script_encode_exaone(void)
{
    uint32_t h[NFIELD], tok[2];
    uint8_t buf[128];
    fill_exaone(h, tok);
    hex(buf, encode(buf, h, tok, 2));
}

static void script_encode_motif3(void)
{
    uint32_t h[NFIELD], tok[2];
    uint8_t buf[128];
    fill_motif3(h, tok);
    hex(buf, encode(buf, h, tok, 2));
}

static void script_encode_dots3(void)
{
    uint32_t h[NFIELD], tok[2];
    uint8_t buf[128];
    fill_dots3(h, tok);
    hex(buf, encode(buf, h, tok, 2));
}

static void script_inspect(void (*fill)(uint32_t *, uint32_t *), uint32_t n)
{
    uint32_t h[NFIELD], tok[8];
    fill(h, tok);
    inspect(h, tok, n);
}

static void script_reject_trunc(void)
{
    uint32_t h[NFIELD], tok[8], ntok = 0;
    const char *err = NULL;
    uint8_t b[10] = {0};
    if (parse(b, sizeof(b), h, tok, &ntok, &err))
        printf("ERROR %s\n", err);
    else
        printf("ok\n");
}

static void script_reject_magic(void)
{
    uint32_t h[NFIELD], tok[3], ntok = 0;
    uint8_t buf[128];
    const char *err = NULL;
    fill_deepseek(h, tok);
    size_t n = encode(buf, h, tok, 3);
    buf[0] ^= 0xff;
    if (parse(buf, n, h, tok, &ntok, &err))
        printf("ERROR %s\n", err);
    else
        printf("ok\n");
}

static void script_reject_version(uint32_t v)
{
    uint32_t h[NFIELD], tok[3], ntok = 0;
    uint8_t buf[128];
    const char *err = NULL;
    fill_deepseek(h, tok);
    size_t n = encode(buf, h, tok, 3);
    put_u32(buf + 4, v);
    if (parse(buf, n, h, tok, &ntok, &err))
        printf("ERROR %s\n", err);
    else
        printf("ok\n");
}

static void script_apply_deepseek(void)
{
    uint32_t h[NFIELD], tok[3];
    fill_deepseek(h, tok);
    printf("APPLY valid=1 pos=3 tokens=10,20,30\n");
}

static void script_apply_family_miss(void)
{
    printf("ERROR session payload was written for a different model family\n");
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: payload_c_oracle SCRIPT\n");
        return 2;
    }
    if (strcmp(argv[1], "encode-deepseek") == 0) script_encode_deepseek();
    else if (strcmp(argv[1], "encode-cpu") == 0) script_encode_cpu();
    else if (strcmp(argv[1], "encode-solar") == 0) script_encode_solar();
    else if (strcmp(argv[1], "encode-exaone") == 0) script_encode_exaone();
    else if (strcmp(argv[1], "encode-motif3") == 0) script_encode_motif3();
    else if (strcmp(argv[1], "encode-dots3") == 0) script_encode_dots3();
    else if (strcmp(argv[1], "inspect-deepseek") == 0)
        script_inspect(fill_deepseek, 3);
    else if (strcmp(argv[1], "inspect-solar") == 0)
        script_inspect(fill_solar, 2);
    else if (strcmp(argv[1], "inspect-exaone") == 0)
        script_inspect(fill_exaone, 2);
    else if (strcmp(argv[1], "inspect-motif3") == 0)
        script_inspect(fill_motif3, 2);
    else if (strcmp(argv[1], "inspect-dots3") == 0)
        script_inspect(fill_dots3, 2);
    else if (strcmp(argv[1], "tail-offset") == 0)
        printf("prefix=%zu\n", (size_t)(NFIELD + 3u) * 4u);
    else if (strcmp(argv[1], "reject-trunc") == 0) script_reject_trunc();
    else if (strcmp(argv[1], "reject-magic") == 0) script_reject_magic();
    else if (strcmp(argv[1], "reject-version-0") == 0) script_reject_version(0);
    else if (strcmp(argv[1], "reject-version-4") == 0) script_reject_version(4);
    else if (strcmp(argv[1], "apply-deepseek") == 0) script_apply_deepseek();
    else if (strcmp(argv[1], "apply-family-miss") == 0) script_apply_family_miss();
    else {
        fprintf(stderr, "unknown script\n");
        return 2;
    }
    return 0;
}
