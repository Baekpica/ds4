/* Generic model-family CUDA primitive checks.
 *
 * These operations are shared by Solar Open 2 and the later EXAONE port:
 * Q8_0 token embedding, sigmoid+biased-top-k routing with unbiased normalized
 * weights, and router-weighted SwiGLU. The references below encode the model
 * equations independently from the CUDA implementation.
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    T_VOCAB = 17,
    T_EMBD = 128,
    T_EMBED_TOKENS = 64,
    T_EXPERT = 320,
    T_USED = 8,
    T_ROUTER_TOKENS = 4,
    T_SWIGLU_TOKENS = 3,
    T_FF = 257,
    T_MOE_EXPERT = 16,
    T_MOE_USED = 8,
    T_MOE_IN = 256,
    T_MOE_OUT = 64,
};

typedef struct {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
} test_block_q3_k;

_Static_assert(sizeof(test_block_q3_k) == 110, "Q3_K block layout");

static int failures;

#define REQUIRE(c, m) do {                                                    \
    if (!(c)) {                                                               \
        fprintf(stderr, "FAIL: %s (line %d)\n", (m), __LINE__);            \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
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
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            while ((mant & 0x0400u) == 0u) { mant <<= 1; exp--; }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float sigmoid_ref(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x))
                     : expf(x) / (1.0f + expf(x));
}

static void compare_f32(const char *label, const float *got, const float *want,
                        size_t n, double atol, double rtol) {
    double max_abs = 0.0, max_rel = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        const double delta = fabs((double)got[i] - want[i]);
        const double scale = fmax(fabs((double)got[i]), fabs((double)want[i]));
        if (delta > max_abs) { max_abs = delta; worst = i; }
        if (scale > 1.0e-12 && delta / scale > max_rel) max_rel = delta / scale;
    }
    const int ok = max_abs <= atol || max_rel <= rtol;
    if (!ok) failures++;
    printf("%-38s abs=%.3e rel=%.3e worst=%zu %s\n",
           label, max_abs, max_rel, worst, ok ? "ok" : "FAIL");
}

static void fill_q8_embedding(unsigned char *table) {
    const uint32_t blocks = T_EMBD / 32u;
    for (uint32_t tok = 0; tok < T_VOCAB; tok++) {
        for (uint32_t block = 0; block < blocks; block++) {
            unsigned char *p = table + ((uint64_t)tok * blocks + block) * 34u;
            const uint16_t d = f32_to_f16(0.03125f * (float)(1u + (tok + block) % 7u));
            memcpy(p, &d, sizeof(d));
            for (uint32_t i = 0; i < 32u; i++) {
                ((int8_t *)(p + 2))[i] =
                    (int8_t)((int)((tok * 29u + block * 11u + i * 3u) % 255u) - 127);
            }
        }
    }
}

static void embedding_ref(float *out, const unsigned char *table, int32_t token) {
    if (token < 0 || token >= T_VOCAB) {
        memset(out, 0, T_EMBD * sizeof(float));
        return;
    }
    const uint32_t blocks = T_EMBD / 32u;
    for (uint32_t d = 0; d < T_EMBD; d++) {
        const unsigned char *p = table +
            ((uint64_t)(uint32_t)token * blocks + (d >> 5)) * 34u;
        uint16_t scale_bits;
        memcpy(&scale_bits, p, sizeof(scale_bits));
        out[d] = f16_to_f32(scale_bits) * (float)((const int8_t *)(p + 2))[d & 31u];
    }
}

static void test_embedding(void *map, uint64_t map_size, uint64_t embed_off) {
    int32_t tokens[T_EMBED_TOKENS];
    for (uint32_t t = 0; t < T_EMBED_TOKENS; t++) {
        tokens[t] = (int32_t)((t * 7u + 3u) % T_VOCAB);
    }
    tokens[T_EMBED_TOKENS - 2] = -1;
    tokens[T_EMBED_TOKENS - 1] = T_VOCAB;

    const uint64_t out_bytes =
        (uint64_t)T_EMBED_TOKENS * T_EMBD * sizeof(float);
    ds4_gpu_tensor *dt = ds4_gpu_tensor_alloc(sizeof(tokens));
    ds4_gpu_tensor *single = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *batch = ds4_gpu_tensor_alloc(out_bytes);
    REQUIRE(dt && single && batch, "embedding tensors");
    REQUIRE(ds4_gpu_tensor_write(dt, 0, tokens, sizeof(tokens)), "write token ids");

    float *want = malloc((size_t)out_bytes);
    float *got_single = malloc((size_t)out_bytes);
    float *got_batch = malloc((size_t)out_bytes);
    REQUIRE(want && got_single && got_batch, "embedding host arrays");
    for (uint32_t t = 0; t < T_EMBED_TOKENS; t++) {
        embedding_ref(want + (size_t)t * T_EMBD,
                      (const unsigned char *)map + embed_off, tokens[t]);
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(
            single, (uint64_t)t * T_EMBD * sizeof(float),
            (uint64_t)T_EMBD * sizeof(float));
        REQUIRE(row, "single embedding view");
        if (tokens[t] >= 0 && tokens[t] < T_VOCAB) {
            REQUIRE(ds4_gpu_embed_token_quant_tensor(
                row, map, map_size, embed_off, 8u, T_VOCAB,
                (uint32_t)tokens[t], T_EMBD), "single Q8 embedding");
        } else {
            REQUIRE(ds4_gpu_tensor_fill_f32(row, 0.0f, T_EMBD),
                    "single invalid embedding zero");
        }
        ds4_gpu_tensor_free(row);
    }
    REQUIRE(ds4_gpu_embed_tokens_quant_tensor(
        batch, dt, map, map_size, embed_off, 8u, T_VOCAB,
        T_EMBED_TOKENS, T_EMBD), "batched Q8 embedding");
    REQUIRE(ds4_gpu_tensor_read(single, 0, got_single, out_bytes),
            "read single embeddings");
    REQUIRE(ds4_gpu_tensor_read(batch, 0, got_batch, out_bytes),
            "read batch embeddings");
    compare_f32("Q8 embedding rows vs CPU", got_single, want,
                (size_t)T_EMBED_TOKENS * T_EMBD, 0.0, 0.0);
    compare_f32("Q8 embedding batch vs CPU", got_batch, want,
                (size_t)T_EMBED_TOKENS * T_EMBD, 0.0, 0.0);

    free(got_batch); free(got_single); free(want);
    ds4_gpu_tensor_free(batch); ds4_gpu_tensor_free(single); ds4_gpu_tensor_free(dt);
}

static void router_ref(int32_t *selected, float *weights,
                       const float *logits, const float *bias, float scale) {
    float probs[T_EXPERT];
    float scores[T_EXPERT];
    for (uint32_t e = 0; e < T_EXPERT; e++) {
        probs[e] = isfinite(logits[e]) ? sigmoid_ref(logits[e]) : 0.0f;
        scores[e] = probs[e] + (isfinite(bias[e]) ? bias[e] : 0.0f);
    }
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < T_USED; slot++) {
        int32_t best = -1;
        float best_score = -INFINITY;
        for (uint32_t e = 0; e < T_EXPERT; e++) {
            if (scores[e] > best_score) {
                best_score = scores[e];
                best = (int32_t)e;
            }
        }
        selected[slot] = best;
        weights[slot] = probs[(uint32_t)best];
        sum += weights[slot];
        scores[(uint32_t)best] = -INFINITY;
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t slot = 0; slot < T_USED; slot++) {
        weights[slot] = weights[slot] / sum * scale;
    }
}

static void test_router(void *map, uint64_t map_size, uint64_t bias_off) {
    const size_t logits_n = (size_t)T_ROUTER_TOKENS * T_EXPERT;
    const size_t out_n = (size_t)T_ROUTER_TOKENS * T_USED;
    float *logits = malloc(logits_n * sizeof(float));
    float *want_w = malloc(out_n * sizeof(float));
    float *got_w = malloc(out_n * sizeof(float));
    int32_t *want_ids = malloc(out_n * sizeof(int32_t));
    int32_t *got_ids = malloc(out_n * sizeof(int32_t));
    REQUIRE(logits && want_w && got_w && want_ids && got_ids, "router host arrays");
    float *bias = (float *)((unsigned char *)map + bias_off);
    for (uint32_t e = 0; e < T_EXPERT; e++) {
        bias[e] = 0.035f * sinf(0.19f * (float)e);
    }
    bias[5] = NAN;
    for (uint32_t t = 0; t < T_ROUTER_TOKENS; t++) {
        float *row = logits + (size_t)t * T_EXPERT;
        for (uint32_t e = 0; e < T_EXPERT; e++) {
            row[e] = -7.0f + 0.001f * (float)((e * 17u + t * 13u) % 101u);
        }
        for (uint32_t rank = 0; rank < T_USED + 3u; rank++) {
            const uint32_t e = (t * 53u + rank * 37u + 11u) % T_EXPERT;
            row[e] = 5.0f - 0.31f * (float)rank;
        }
        if (t == T_ROUTER_TOKENS - 1u) {
            for (uint32_t e = 0; e < T_EXPERT; e++) row[e] = NAN;
            row[17] = INFINITY;
            row[29] = -INFINITY;
        }
        router_ref(want_ids + (size_t)t * T_USED,
                   want_w + (size_t)t * T_USED, row, bias, 2.5f);
    }

    ds4_gpu_tensor *dl = ds4_gpu_tensor_alloc(logits_n * sizeof(float));
    ds4_gpu_tensor *di = ds4_gpu_tensor_alloc(out_n * sizeof(int32_t));
    ds4_gpu_tensor *dw = ds4_gpu_tensor_alloc(out_n * sizeof(float));
    REQUIRE(dl && di && dw, "router tensors");
    REQUIRE(ds4_gpu_tensor_write(dl, 0, logits, logits_n * sizeof(float)),
            "write router logits");
    REQUIRE(ds4_gpu_sigmoid_topk_router_tensor(
        di, dw, dl, map, map_size, bias_off, 1,
        T_EXPERT, T_USED, T_ROUTER_TOKENS, 2.5f), "sigmoid top-k router");
    REQUIRE(ds4_gpu_tensor_read(di, 0, got_ids, out_n * sizeof(int32_t)),
            "read router ids");
    REQUIRE(ds4_gpu_tensor_read(dw, 0, got_w, out_n * sizeof(float)),
            "read router weights");
    size_t bad_ids = 0;
    for (size_t i = 0; i < out_n; i++) if (got_ids[i] != want_ids[i]) bad_ids++;
    printf("%-38s mismatches=%zu %s\n", "sigmoid router selected ids", bad_ids,
           bad_ids ? "FAIL" : "ok");
    if (bad_ids) failures++;
    compare_f32("sigmoid router unbiased weights", got_w, want_w, out_n,
                2.0e-6, 3.0e-6);

    ds4_gpu_tensor_free(dw); ds4_gpu_tensor_free(di); ds4_gpu_tensor_free(dl);
    free(got_ids); free(want_ids); free(got_w); free(want_w); free(logits);
}

static void test_weighted_swiglu(void) {
    const size_t slots = (size_t)T_SWIGLU_TOKENS * T_USED;
    const size_t count = slots * T_FF;
    float *gate = malloc(count * sizeof(float));
    float *up = malloc(count * sizeof(float));
    float *weights = malloc(slots * sizeof(float));
    float *want = malloc(count * sizeof(float));
    float *got = malloc(count * sizeof(float));
    REQUIRE(gate && up && weights && want && got, "SwiGLU host arrays");
    for (size_t slot = 0; slot < slots; slot++) {
        weights[slot] = 0.05f + 0.01f * (float)((slot * 7u) % 19u);
    }
    for (size_t i = 0; i < count; i++) {
        gate[i] = 3.0f * sinf(0.013f * (float)(i + 1u));
        up[i] = 2.5f * cosf(0.017f * (float)(i + 3u));
        want[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i] * weights[i / T_FF];
    }
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *du = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *dw = ds4_gpu_tensor_alloc(slots * sizeof(float));
    ds4_gpu_tensor *dm = ds4_gpu_tensor_alloc(count * sizeof(float));
    REQUIRE(dg && du && dw && dm, "SwiGLU tensors");
    REQUIRE(ds4_gpu_tensor_write(dg, 0, gate, count * sizeof(float)), "write gate");
    REQUIRE(ds4_gpu_tensor_write(du, 0, up, count * sizeof(float)), "write up");
    REQUIRE(ds4_gpu_tensor_write(dw, 0, weights, slots * sizeof(float)), "write weights");
    REQUIRE(ds4_gpu_swiglu_weighted_tensor(dm, dg, du, dw, T_FF, count),
            "weighted SwiGLU");
    REQUIRE(ds4_gpu_tensor_read(dm, 0, got, count * sizeof(float)),
            "read weighted SwiGLU");
    compare_f32("router-weighted SwiGLU", got, want, count, 1.0e-5, 1.0e-5);

    ds4_gpu_tensor_free(dm); ds4_gpu_tensor_free(dw);
    ds4_gpu_tensor_free(du); ds4_gpu_tensor_free(dg);
    free(got); free(want); free(weights); free(up); free(gate);
}

static uint32_t next_u32(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_q3_weights(test_block_q3_k *weights) {
    uint32_t state = 0x51a7c0deu;
    const size_t blocks = (size_t)T_MOE_EXPERT * T_MOE_OUT;
    for (size_t b = 0; b < blocks; b++) {
        for (size_t i = 0; i < sizeof(weights[b].hmask); i++) {
            weights[b].hmask[i] = (uint8_t)(next_u32(&state) >> 24);
        }
        for (size_t i = 0; i < sizeof(weights[b].qs); i++) {
            weights[b].qs[i] = (uint8_t)(next_u32(&state) >> 24);
        }
        for (size_t i = 0; i < sizeof(weights[b].scales); i++) {
            weights[b].scales[i] = (uint8_t)(next_u32(&state) >> 24);
        }
        weights[b].d = f32_to_f16(0.006f + 0.001f * (float)(b % 13u));
    }
}

static void dequant_q3_block(const test_block_q3_k *block, float *out) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4] = {0, 0, 0, 0};
    memcpy(aux, block->scales, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t *scales = (const int8_t *)aux;
    const float d_all = f16_to_f32(block->d);
    const uint8_t *q = block->qs;
    uint8_t mask = 1u;
    int is = 0;
    size_t oi = 0;
    for (uint32_t half = 0; half < 2u; half++) {
        (void)half;
        int shift = 0;
        for (uint32_t group = 0; group < 4u; group++) {
            (void)group;
            float scale = d_all * (float)(scales[is++] - 32);
            for (uint32_t i = 0; i < 16u; i++) {
                out[oi++] = scale * (float)((int)((q[i] >> shift) & 3u) -
                    ((block->hmask[i] & mask) ? 0 : 4));
            }
            scale = d_all * (float)(scales[is++] - 32);
            for (uint32_t i = 0; i < 16u; i++) {
                out[oi++] = scale * (float)((int)((q[i + 16u] >> shift) & 3u) -
                    ((block->hmask[i + 16u] & mask) ? 0 : 4));
            }
            shift += 2;
            mask <<= 1;
        }
        q += 32;
    }
}

static void test_q3_routed_matmul(void *map, uint64_t map_size,
                                  uint64_t weight_off, uint64_t weight_bytes) {
    const int32_t ids[T_MOE_USED] = {7, 3, 15, 0, 11, 5, 13, 9};
    float x[T_MOE_IN];
    float want[T_MOE_USED * T_MOE_OUT];
    float got[T_MOE_USED * T_MOE_OUT];
    for (uint32_t i = 0; i < T_MOE_IN; i++) {
        x[i] = 0.65f * sinf(0.031f * (float)(i + 1u)) +
               0.25f * cosf(0.017f * (float)(i + 3u));
    }
    const test_block_q3_k *weights =
        (const test_block_q3_k *)((const unsigned char *)map + weight_off);
    for (uint32_t slot = 0; slot < T_MOE_USED; slot++) {
        const uint32_t expert = (uint32_t)ids[slot];
        for (uint32_t row = 0; row < T_MOE_OUT; row++) {
            float deq[T_MOE_IN];
            dequant_q3_block(weights + (size_t)expert * T_MOE_OUT + row, deq);
            float sum = 0.0f;
            for (uint32_t k = 0; k < T_MOE_IN; k++) sum += deq[k] * x[k];
            want[(size_t)slot * T_MOE_OUT + row] = sum;
        }
    }

    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *di = ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(sizeof(got));
    REQUIRE(dx && di && dy, "Q3 routed matmul tensors");
    REQUIRE(ds4_gpu_tensor_write(dx, 0, x, sizeof(x)), "write Q3 activation");
    REQUIRE(ds4_gpu_tensor_write(di, 0, ids, sizeof(ids)), "write Q3 ids");
    REQUIRE(ds4_gpu_routed_matmul_tensor(
        dy, dx, di, map, map_size, weight_off, weight_bytes, 11u,
        T_MOE_IN, T_MOE_OUT, T_MOE_EXPERT, 1u, T_MOE_USED),
        "Q3 routed matmul dispatch");
    REQUIRE(ds4_gpu_tensor_read(dy, 0, got, sizeof(got)), "read Q3 output");

    double err2 = 0.0, ref2 = 0.0, got2 = 0.0, dot = 0.0;
    for (size_t i = 0; i < T_MOE_USED * T_MOE_OUT; i++) {
        const double e = (double)got[i] - want[i];
        err2 += e * e;
        ref2 += (double)want[i] * want[i];
        got2 += (double)got[i] * got[i];
        dot += (double)got[i] * want[i];
    }
    const double rel_rms = sqrt(err2 / ref2);
    const double one_minus_cos = 1.0 - dot / sqrt(got2 * ref2);
    const int ok = rel_rms <= 2.0e-2 && one_minus_cos <= 2.0e-2;
    printf("%-38s rel_rms=%.3e 1-cos=%.3e %s\n",
           "Q3_K routed matmul vs CPU", rel_rms, one_minus_cos,
           ok ? "ok" : "FAIL");
    if (!ok) failures++;
    ds4_gpu_tensor_free(dy); ds4_gpu_tensor_free(di); ds4_gpu_tensor_free(dx);
}

int main(void) {
    REQUIRE(ds4_gpu_init(), "CUDA init");
    const uint64_t embed_off = 4096u;
    const uint64_t embed_bytes = (uint64_t)T_VOCAB * (T_EMBD / 32u) * 34u;
    const uint64_t bias_off = (embed_off + embed_bytes + 4095u) & ~4095ull;
    const uint64_t q3_off = bias_off + 4096u;
    const uint64_t q3_bytes =
        (uint64_t)T_MOE_EXPERT * T_MOE_OUT * sizeof(test_block_q3_k);
    const uint64_t map_size = (q3_off + q3_bytes + 4095u) & ~4095ull;
    void *map = NULL;
    REQUIRE(posix_memalign(&map, 4096u, (size_t)map_size) == 0, "model map alloc");
    memset(map, 0, (size_t)map_size);
    fill_q8_embedding((unsigned char *)map + embed_off);
    fill_q3_weights((test_block_q3_k *)((unsigned char *)map + q3_off));
    REQUIRE(ds4_gpu_set_model_map(map, map_size), "model map registration");

    puts("== generic model-family CUDA primitives ==");
    test_embedding(map, map_size, embed_off);
    test_router(map, map_size, bias_off);
    test_weighted_swiglu();
    test_q3_routed_matmul(map, map_size, q3_off, q3_bytes);

    ds4_gpu_unregister_model_map(map);
    free(map);
    ds4_gpu_cleanup();
    puts(failures ? "model-family primitive checks FAILED"
                  : "all model-family primitive checks passed");
    return failures ? 1 : 0;
}
