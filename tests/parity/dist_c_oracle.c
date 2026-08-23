/* C DS4D wire oracle. Structs and htonl helpers are copied from
 * ds4_distributed.c at v0.6.3-dfm so Rust codecs can compare bytes
 * without linking the engine. */

#include <arpa/inet.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_DIST_MAGIC 0x44533444u
#define DS4_DIST_TOKEN_HASH_INIT 1469598103934665603ull
#define DS4_DIST_TOKEN_HASH_PRIME 1099511628211ull

typedef struct { uint32_t magic, type, bytes; } ds4_dist_frame_header;
typedef struct {
    uint32_t model_id, quant_bits, layer_start, layer_end, has_output, has_hidden,
             ctx_size, n_layers, listen_port, model_name_len;
} ds4_dist_hello_fixed;
typedef struct {
    uint32_t model_id, session_hi, session_lo, request_hi, request_lo,
             prefix_hash_hi, prefix_hash_lo, result_hash_hi, result_hash_lo,
             pos0, n_tokens, layer_start, layer_end, flags, token_bytes,
             input_hc_bytes, input_hc_bits, route_count, route_index, route_bytes;
} ds4_dist_work_fixed;
typedef struct {
    uint32_t host_len, port, layer_start, layer_end, flags;
} ds4_dist_route_fixed;
typedef struct { uint32_t kind, host_len, port; } ds4_dist_route_return_fixed;
typedef struct {
    uint32_t request_hi, request_lo, result_hash_hi, result_hash_lo, status,
             result_kind, telemetry_count, telemetry_bytes, payload_bytes, payload_bits;
} ds4_dist_result_fixed;
typedef struct {
    uint32_t layer_start, layer_end, route_index, pos0, n_tokens, eval_usec,
             downstream_wait_usec, forward_send_usec, input_bytes, output_bytes;
} ds4_dist_telemetry_fixed;
typedef struct {
    uint32_t model_id, session_hi, session_lo, request_hi, request_lo,
             token_hash_hi, token_hash_lo, token_count, layer_start, layer_end;
} ds4_dist_snapshot_req_fixed;
typedef struct {
    uint32_t model_id, session_hi, session_lo, request_hi, request_lo,
             token_hash_hi, token_hash_lo, token_count, layer_start, layer_end,
             payload_hi, payload_lo, status, token_bytes, message_bytes;
} ds4_dist_snapshot_begin_fixed;
typedef struct { uint32_t request_hi, request_lo, chunk_bytes; } ds4_dist_snapshot_chunk_fixed;
typedef struct { uint32_t request_hi, request_lo, status, message_bytes; } ds4_dist_snapshot_done_fixed;

static void die(const char *m) { fprintf(stderr, "dist_c_oracle: %s\n", m); exit(2); }

static void print_hex(const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
}

static uint32_t need_u32(const char *s) { return (uint32_t)strtoul(s, NULL, 0); }

static void dist_hello_to_wire(ds4_dist_hello_fixed *h) {
    h->model_id = htonl(h->model_id); h->quant_bits = htonl(h->quant_bits);
    h->layer_start = htonl(h->layer_start); h->layer_end = htonl(h->layer_end);
    h->has_output = htonl(h->has_output); h->has_hidden = htonl(h->has_hidden);
    h->ctx_size = htonl(h->ctx_size); h->n_layers = htonl(h->n_layers);
    h->listen_port = htonl(h->listen_port); h->model_name_len = htonl(h->model_name_len);
}
static void dist_work_to_wire(ds4_dist_work_fixed *w) {
    uint32_t *p = (uint32_t *)w;
    for (size_t i = 0; i < sizeof(*w) / 4; i++) p[i] = htonl(p[i]);
}
static void dist_route_to_wire(ds4_dist_route_fixed *r) {
    r->host_len = htonl(r->host_len); r->port = htonl(r->port);
    r->layer_start = htonl(r->layer_start); r->layer_end = htonl(r->layer_end);
    r->flags = htonl(r->flags);
}
static void dist_route_return_to_wire(ds4_dist_route_return_fixed *r) {
    r->kind = htonl(r->kind); r->host_len = htonl(r->host_len); r->port = htonl(r->port);
}
static void dist_result_to_wire(ds4_dist_result_fixed *r) {
    uint32_t *p = (uint32_t *)r;
    for (size_t i = 0; i < sizeof(*r) / 4; i++) p[i] = htonl(p[i]);
}
static void __attribute__((unused)) dist_telemetry_to_wire(ds4_dist_telemetry_fixed *t) {
    uint32_t *p = (uint32_t *)t;
    for (size_t i = 0; i < sizeof(*t) / 4; i++) p[i] = htonl(p[i]);
}
static void __attribute__((unused)) dist_snapshot_req_to_wire(ds4_dist_snapshot_req_fixed *s) {
    uint32_t *p = (uint32_t *)s;
    for (size_t i = 0; i < sizeof(*s) / 4; i++) p[i] = htonl(p[i]);
}
static void __attribute__((unused)) dist_snapshot_begin_to_wire(ds4_dist_snapshot_begin_fixed *s) {
    uint32_t *p = (uint32_t *)s;
    for (size_t i = 0; i < sizeof(*s) / 4; i++) p[i] = htonl(p[i]);
}
static void __attribute__((unused)) dist_snapshot_chunk_to_wire(ds4_dist_snapshot_chunk_fixed *s) {
    s->request_hi = htonl(s->request_hi); s->request_lo = htonl(s->request_lo);
    s->chunk_bytes = htonl(s->chunk_bytes);
}
static void __attribute__((unused)) dist_snapshot_done_to_wire(ds4_dist_snapshot_done_fixed *s) {
    s->request_hi = htonl(s->request_hi); s->request_lo = htonl(s->request_lo);
    s->status = htonl(s->status); s->message_bytes = htonl(s->message_bytes);
}

static uint64_t dist_token_hash_update(uint64_t h, int token) {
    uint32_t t = (uint32_t)token;
    for (int i = 0; i < 4; i++) {
        h ^= (uint64_t)((t >> (i * 8)) & 0xffu);
        h *= DS4_DIST_TOKEN_HASH_PRIME;
    }
    return h;
}

static uint16_t dist_f32_to_f16(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) return (uint16_t)(sign | 0x7e00u);
        return (uint16_t)(sign | 0x7c00u);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

static uint8_t dist_f32_to_f8_e4m3(float f) {
    const uint8_t sign = signbit(f) ? 0x80u : 0u;
    float a = fabsf(f);
    if (a == 0.0f) return sign;
    if (!isfinite(a) || a >= 240.0f) return (uint8_t)(sign | 0x77u);
    if (a < 0.001953125f) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }
    int exp2 = 0;
    (void)frexpf(a, &exp2);
    int exp = exp2 - 1 + 7;
    if (exp <= 0) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }
    float base = ldexpf(1.0f, exp2 - 1);
    int mant = (int)floorf(((a / base) - 1.0f) * 8.0f + 0.5f);
    if (mant >= 8) { mant = 0; exp++; }
    if (exp >= 15) return (uint8_t)(sign | 0x77u);
    return (uint8_t)(sign | (uint8_t)(exp << 3) | (uint8_t)mant);
}

static float bits_to_f32(const char *hex) {
    uint32_t bits = (uint32_t)strtoul(hex, NULL, 16);
    float f; memcpy(&f, &bits, sizeof(f));
    return f;
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage");
    const char *cmd = argv[1];
    if (!strcmp(cmd, "sizes")) {
        printf("frame %zu hello %zu work %zu route %zu ret %zu result %zu tel %zu sreq %zu sbeg %zu schunk %zu sdone %zu\n",
               sizeof(ds4_dist_frame_header), sizeof(ds4_dist_hello_fixed),
               sizeof(ds4_dist_work_fixed), sizeof(ds4_dist_route_fixed),
               sizeof(ds4_dist_route_return_fixed), sizeof(ds4_dist_result_fixed),
               sizeof(ds4_dist_telemetry_fixed), sizeof(ds4_dist_snapshot_req_fixed),
               sizeof(ds4_dist_snapshot_begin_fixed), sizeof(ds4_dist_snapshot_chunk_fixed),
               sizeof(ds4_dist_snapshot_done_fixed));
    } else if (!strcmp(cmd, "frame")) {
        if (argc < 4) die("frame TYPE BYTES");
        ds4_dist_frame_header h = { htonl(DS4_DIST_MAGIC), htonl(need_u32(argv[2])), htonl(need_u32(argv[3])) };
        print_hex(&h, sizeof(h));
    } else if (!strcmp(cmd, "hello")) {
        if (argc < 12) die("hello fields... NAME");
        ds4_dist_hello_fixed h = {
            need_u32(argv[2]), need_u32(argv[3]), need_u32(argv[4]), need_u32(argv[5]),
            need_u32(argv[6]), need_u32(argv[7]), need_u32(argv[8]), need_u32(argv[9]),
            need_u32(argv[10]), (uint32_t)strlen(argv[11])
        };
        const char *name = argv[11];
        dist_hello_to_wire(&h);
        print_hex(&h, sizeof(h));
        print_hex(name, strlen(name));
    } else if (!strcmp(cmd, "work")) {
        if (argc < 22) die("work 20 fields");
        ds4_dist_work_fixed w;
        uint32_t *p = (uint32_t *)&w;
        for (int i = 0; i < 20; i++) p[i] = need_u32(argv[2 + i]);
        dist_work_to_wire(&w);
        print_hex(&w, sizeof(w));
    } else if (!strcmp(cmd, "tokens")) {
        for (int i = 2; i < argc; i++) {
            uint32_t t = htonl((uint32_t)atoi(argv[i]));
            print_hex(&t, 4);
        }
    } else if (!strcmp(cmd, "token-hash")) {
        uint64_t h = DS4_DIST_TOKEN_HASH_INIT;
        for (int i = 2; i < argc; i++) h = dist_token_hash_update(h, atoi(argv[i]));
        printf("%016llx", (unsigned long long)h);
    } else if (!strcmp(cmd, "route")) {
        if (argc < 7) die("route HOST PORT LS LE FLAGS");
        ds4_dist_route_fixed r = {
            (uint32_t)strlen(argv[2]), need_u32(argv[3]), need_u32(argv[4]),
            need_u32(argv[5]), need_u32(argv[6])
        };
        dist_route_to_wire(&r);
        print_hex(&r, sizeof(r));
        print_hex(argv[2], strlen(argv[2]));
    } else if (!strcmp(cmd, "route-return")) {
        if (argc < 5) die("route-return KIND HOST PORT");
        ds4_dist_route_return_fixed r = { need_u32(argv[2]), (uint32_t)strlen(argv[3]), need_u32(argv[4]) };
        dist_route_return_to_wire(&r);
        print_hex(&r, sizeof(r));
        print_hex(argv[3], strlen(argv[3]));
    } else if (!strcmp(cmd, "result")) {
        if (argc < 12) die("result 10 fields");
        ds4_dist_result_fixed r;
        uint32_t *p = (uint32_t *)&r;
        for (int i = 0; i < 10; i++) p[i] = need_u32(argv[2 + i]);
        dist_result_to_wire(&r);
        print_hex(&r, sizeof(r));
    } else if (!strcmp(cmd, "f16")) {
        if (argc < 3) die("f16 F32HEX");
        uint16_t h = dist_f32_to_f16(bits_to_f32(argv[2]));
        print_hex(&h, 2);
    } else if (!strcmp(cmd, "f8")) {
        if (argc < 3) die("f8 F32HEX");
        uint8_t h = dist_f32_to_f8_e4m3(bits_to_f32(argv[2]));
        print_hex(&h, 1);
    } else {
        die("unknown command");
    }
    return 0;
}
