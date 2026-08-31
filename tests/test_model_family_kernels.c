/* Generic model-family CUDA primitive checks.
 *
 * These operations are shared by Solar Open 2 and the later EXAONE port:
 * Q8_0 token embedding, sigmoid+biased-top-k routing with unbiased normalized
 * weights, and router-weighted SwiGLU. The references below encode the model
 * equations independently from the CUDA implementation.
 */
#include "ds4_gpu.h"

#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

_Static_assert(sizeof(((ds4_gpu_tensor_record *)0)->dims) / sizeof(uint64_t) == 8,
               "derived catalog must preserve every GGUF dimension");

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
    T_IQ2_IN = 1024,
    /* Solar Open2's routed expert intermediate width. Keeping the production
     * 1280 here exercises all ten k128 D4 blocks, including the final partial
     * four-warp emitter CTA, instead of only the first two blocks. */
    T_IQ2_OUT = 1280,
    /* 512 * top-8 reaches the production D2R threshold while staying above
     * the handoff's conservative prefill-only floor. It exercises the aligned
     * routed gate/up pair path, including ragged expert buckets. */
    T_IQ2_TOKENS = 512,
    /* 512 also satisfies the dense MMQ row-padding contract, so the test's
     * verify mode byte-diffs the producer mirror against the real reference
     * quantizer instead of declining the diagnostic. */
    T_NORM_DIM = 512,
    T_NORM_ROWS = 64,
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
    size_t first_nonfinite = 0, nonfinite = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(got[i]) || !isfinite(want[i])) {
            if (nonfinite == 0) first_nonfinite = i;
            nonfinite++;
            continue;
        }
        const double delta = fabs((double)got[i] - want[i]);
        const double scale = fmax(fabs((double)got[i]), fabs((double)want[i]));
        if (delta > max_abs) { max_abs = delta; worst = i; }
        if (scale > 1.0e-12 && delta / scale > max_rel) max_rel = delta / scale;
    }
    const int ok = nonfinite == 0 && (max_abs <= atol || max_rel <= rtol);
    if (!ok) failures++;
    printf("%-38s abs=%.3e rel=%.3e worst=%zu nonfinite=%zu",
           label, max_abs, max_rel, worst, nonfinite);
    if (nonfinite != 0) printf(" first_nonfinite=%zu", first_nonfinite);
    printf(" %s\n", ok ? "ok" : "FAIL");
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

static void test_weighted_swiglu_q8_d4_bytes(void) {
    enum { NA = 64, K = 1280 };
    const size_t count = (size_t)NA * K;
    const size_t q8_bytes = (count / 128u) * 144u;
    float *gate = malloc(count * sizeof(float));
    float *up = malloc(count * sizeof(float));
    float weights[NA];
    int32_t ids_dst[NA];
    unsigned char *emit = malloc(q8_bytes);
    unsigned char *ref = malloc(q8_bytes);
    REQUIRE(gate && up && emit && ref, "D4 parity host arrays");
    for (int p = 0; p < NA; p++) {
        ids_dst[p] = (int32_t)((p * 37) % NA);
        weights[p] = (p & 1) ? -0.07f * (float)(p + 1)
                             : 0.03f * (float)(p + 1);
    }
    for (size_t i = 0; i < count; i++) {
        gate[i] = 2.1f * sinf(0.011f * (float)(i + 1u));
        up[i] = 1.7f * cosf(0.019f * (float)(i + 7u));
    }
    gate[3] = NAN; up[37] = INFINITY; gate[95] = -INFINITY;
    /* One complete 32-value D4 subgroup is exactly zero.  This fixes the
     * canonical 127/amax infinity path (including its byte representation)
     * as part of the fused-emitter parity contract. */
    const int zero_pair = ids_dst[1];
    for (int k = 64; k < 96; k++) {
        gate[(size_t)zero_pair * K + (size_t)k] = 0.0f;
        up[(size_t)zero_pair * K + (size_t)k] = 0.0f;
    }
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *du = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *dw = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *dm = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *di = ds4_gpu_tensor_alloc(sizeof(ids_dst));
    ds4_gpu_tensor *de = ds4_gpu_tensor_alloc(q8_bytes);
    ds4_gpu_tensor *dr = ds4_gpu_tensor_alloc(q8_bytes);
    REQUIRE(dg && du && dw && dm && di && de && dr, "D4 parity tensors");
    REQUIRE(ds4_gpu_tensor_write(dg, 0, gate, count * sizeof(float)),
            "write D4 gate");
    REQUIRE(ds4_gpu_tensor_write(du, 0, up, count * sizeof(float)),
            "write D4 up");
    REQUIRE(ds4_gpu_tensor_write(dw, 0, weights, sizeof(weights)),
            "write D4 router weights");
    REQUIRE(ds4_gpu_tensor_write(di, 0, ids_dst, sizeof(ids_dst)),
            "write D4 ids_dst");
    /* Generate the reference mid with the production GPU SwiGLU kernel;
     * host expf is not byte-equivalent to device fast-math. */
    REQUIRE(ds4_gpu_swiglu_weighted_tensor(dm, dg, du, dw, K, count),
            "production GPU weighted SwiGLU reference");
    REQUIRE(ds4_gpu_swiglu_weighted_q8_d4_emit_test(
                de, dg, du, dw, di, K, NA), "D4 fused emit");
    REQUIRE(ds4_gpu_q3_quantize_ref_test(dr, dm, di, K, NA),
            "canonical Q3 D4 reference quantize");
    REQUIRE(ds4_gpu_tensor_read(de, 0, emit, q8_bytes), "read D4 emit");
    REQUIRE(ds4_gpu_tensor_read(dr, 0, ref, q8_bytes), "read D4 ref");
    const int same = memcmp(emit, ref, q8_bytes) == 0;
    printf("%-38s %s\n", "weighted SwiGLU Q8 D4 bytes",
           same ? "ok" : "FAIL");
    if (!same) failures++;
    ds4_gpu_tensor_free(dr); ds4_gpu_tensor_free(de); ds4_gpu_tensor_free(di);
    ds4_gpu_tensor_free(dm); ds4_gpu_tensor_free(dw); ds4_gpu_tensor_free(du);
    ds4_gpu_tensor_free(dg);
    free(ref); free(emit); free(up); free(gate);
}

static void test_q3_worklist_index_bounds(void) {
    /* Solar production down: [4096,1280], 4096 tokens * top-8, 320 experts. */
    REQUIRE(ds4_gpu_q3_worklist_preflight_test(
                4096, 1280, 4096ll * 8ll, 320, 5, 4096ll * 5ll),
            "Q3 worklist accepts Solar production geometry");
    REQUIRE(!ds4_gpu_q3_worklist_preflight_test(
                128, 256, 1, 2, 1, INT_MAX),
            "Q3 worklist rejects weight int-offset overflow");
    REQUIRE(!ds4_gpu_q3_worklist_preflight_test(
                128, 1280, 10000000ll, 1, 5, 640),
            "Q3 worklist rejects Q8 int-span overflow");
    REQUIRE(!ds4_gpu_q3_worklist_preflight_test(
                128, 256, 20000000ll, 1, 1, 128),
            "Q3 worklist rejects destination int-span overflow");
    printf("%-38s ok\n", "Q3 worklist signed-index bounds");
}

static void test_rms_norm_q8_producer(void *map, uint64_t map_size,
                                      uint64_t norm_off) {
    const size_t count = (size_t)T_NORM_ROWS * T_NORM_DIM;
    float *x = malloc(count * sizeof(*x));
    float *want = malloc(count * sizeof(*want));
    float *got = malloc(count * sizeof(*got));
    float *weight = (float *)((unsigned char *)map + norm_off);
    REQUIRE(x && want && got, "RMS Q8 producer host arrays");
    for (uint32_t i = 0; i < T_NORM_DIM; i++) {
        weight[i] = 0.75f + 0.003f * (float)(i % 67u);
    }
    for (uint32_t row = 0; row < T_NORM_ROWS; row++) {
        double sum = 0.0;
        for (uint32_t i = 0; i < T_NORM_DIM; i++) {
            const size_t at = (size_t)row * T_NORM_DIM + i;
            x[at] = 1.2f * sinf(0.007f * (float)(at + 1u)) +
                    0.4f * cosf(0.019f * (float)(at + 5u));
            sum += (double)x[at] * x[at];
        }
        const float scale = 1.0f /
            sqrtf((float)(sum / (double)T_NORM_DIM) + 1.0e-6f);
        for (uint32_t i = 0; i < T_NORM_DIM; i++) {
            const size_t at = (size_t)row * T_NORM_DIM + i;
            want[at] = x[at] * scale * weight[i];
        }
    }

    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(count * sizeof(float));
    REQUIRE(dx && dy, "RMS Q8 producer tensors");
    REQUIRE(ds4_gpu_tensor_write(dx, 0, x, count * sizeof(float)),
            "write RMS Q8 producer input");
    REQUIRE(ds4_gpu_rms_norm_weight_rows_q8_tensor(
                dy, dx, map, map_size, norm_off,
                T_NORM_DIM, T_NORM_ROWS, 1.0e-6f),
            "RMS norm with Q8 producer");
    REQUIRE(ds4_gpu_tensor_read(dy, 0, got, count * sizeof(float)),
            "read RMS Q8 producer output");
    compare_f32("RMS norm Q8 producer vs CPU", got, want, count,
                3.0e-6, 3.0e-6);

    ds4_gpu_tensor_free(dy); ds4_gpu_tensor_free(dx);
    free(got); free(want); free(x);
}

static uint32_t next_u32(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_q3_weights(test_block_q3_k *weights, size_t blocks,
                            uint32_t state) {
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

static void fill_iq2_weights(unsigned char *weights, uint64_t blocks,
                             uint32_t seed) {
    for (uint64_t b = 0; b < blocks; b++) {
        unsigned char *block = weights + b * 66u;
        const uint16_t d = f32_to_f16(0.003f +
            0.0002f * (float)(b % 11u));
        memcpy(block, &d, sizeof(d));
        for (uint32_t i = 0; i < 64u; i++) {
            block[2u + i] = (uint8_t)(next_u32(&seed) >> 24);
        }
    }
}

static void build_iq2_artifacts(void *map, uint64_t map_size,
                                uint64_t gate_off, uint64_t up_off,
                                uint64_t weight_bytes) {
    const char gate_name[] = "blk.4.ffn_gate_exps.weight";
    const char up_name[] = "blk.4.ffn_up_exps.weight";
    ds4_gpu_tensor_record records[2] = {
        {
            .name = gate_name, .name_len = sizeof(gate_name) - 1u,
            .type = 16u, .ndim = 3u,
            .dims = {T_IQ2_IN, T_IQ2_OUT, T_MOE_EXPERT, 0u},
            .offset = gate_off, .bytes = weight_bytes,
        },
        {
            .name = up_name, .name_len = sizeof(up_name) - 1u,
            .type = 16u, .ndim = 3u,
            .dims = {T_IQ2_IN, T_IQ2_OUT, T_MOE_EXPERT, 0u},
            .offset = up_off, .bytes = weight_bytes,
        },
    };
    REQUIRE(ds4_gpu_build_derived_artifacts_from_records(
                map, map_size, records, 2u) == 2,
            "build IQ2 artifacts from merged records");
    REQUIRE(ds4_gpu_model_map_replacements_complete(map),
            "IQ2 replacement set complete");
}

static void test_iq2_aligned_pair(void *map, uint64_t map_size,
                                  uint64_t gate_off, uint64_t up_off,
                                  uint64_t weight_bytes) {
    const size_t x_n = (size_t)T_IQ2_TOKENS * T_IQ2_IN;
    const size_t ids_n = (size_t)T_IQ2_TOKENS * T_MOE_USED;
    const size_t out_n = ids_n * T_IQ2_OUT;
    float *x = malloc(x_n * sizeof(*x));
    int32_t *ids = malloc(ids_n * sizeof(*ids));
    float *raw_gate = malloc(out_n * sizeof(*raw_gate));
    float *raw_up = malloc(out_n * sizeof(*raw_up));
    float *aligned_gate = malloc(out_n * sizeof(*aligned_gate));
    float *aligned_up = malloc(out_n * sizeof(*aligned_up));
    REQUIRE(x && ids && raw_gate && raw_up && aligned_gate && aligned_up,
            "IQ2 pair host arrays");
    for (size_t i = 0; i < x_n; i++) {
        x[i] = 0.4f * sinf(0.013f * (float)(i + 1u)) +
               0.2f * cosf(0.031f * (float)(i + 7u));
    }
    for (size_t i = 0; i < ids_n; i++) {
        ids[i] = (int32_t)((i * 7u + i / T_MOE_USED * 3u) % T_MOE_EXPERT);
    }

    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(x_n * sizeof(*x));
    ds4_gpu_tensor *di = ds4_gpu_tensor_alloc(ids_n * sizeof(*ids));
    ds4_gpu_tensor *dg_raw = ds4_gpu_tensor_alloc(out_n * sizeof(float));
    ds4_gpu_tensor *du_raw = ds4_gpu_tensor_alloc(out_n * sizeof(float));
    ds4_gpu_tensor *dg_al = ds4_gpu_tensor_alloc(out_n * sizeof(float));
    ds4_gpu_tensor *du_al = ds4_gpu_tensor_alloc(out_n * sizeof(float));
    REQUIRE(dx && di && dg_raw && du_raw && dg_al && du_al,
            "IQ2 pair device tensors");
    REQUIRE(ds4_gpu_tensor_write(dx, 0, x, x_n * sizeof(*x)),
            "write IQ2 activation");
    REQUIRE(ds4_gpu_tensor_write(di, 0, ids, ids_n * sizeof(*ids)),
            "write IQ2 ids");
    REQUIRE(ds4_gpu_routed_matmul_tensor(
        dg_raw, dx, di, map, map_size, gate_off, weight_bytes, 16u,
        T_IQ2_IN, T_IQ2_OUT, T_MOE_EXPERT, T_IQ2_TOKENS, T_MOE_USED),
        "raw IQ2 gate dispatch");
    REQUIRE(ds4_gpu_routed_matmul_tensor(
        du_raw, dx, di, map, map_size, up_off, weight_bytes, 16u,
        T_IQ2_IN, T_IQ2_OUT, T_MOE_EXPERT, T_IQ2_TOKENS, T_MOE_USED),
        "raw IQ2 up dispatch");
    REQUIRE(ds4_gpu_tensor_read(dg_raw, 0, raw_gate,
                                out_n * sizeof(float)), "read raw IQ2 gate");
    REQUIRE(ds4_gpu_tensor_read(du_raw, 0, raw_up,
                                out_n * sizeof(float)), "read raw IQ2 up");

    build_iq2_artifacts(map, map_size, gate_off, up_off, weight_bytes);
    REQUIRE(ds4_gpu_routed_gate_up_tensor(
        dg_al, du_al, dx, di, map, map_size,
        gate_off, weight_bytes, up_off, weight_bytes, 16u,
        T_IQ2_IN, T_IQ2_OUT, T_MOE_EXPERT,
        T_IQ2_TOKENS, T_MOE_USED),
        "aligned IQ2 pair dispatch");
    REQUIRE(ds4_gpu_tensor_read(dg_al, 0, aligned_gate,
                                out_n * sizeof(float)), "read aligned IQ2 gate");
    REQUIRE(ds4_gpu_tensor_read(du_al, 0, aligned_up,
                                out_n * sizeof(float)), "read aligned IQ2 up");
    compare_f32("IQ2 aligned pair gate vs raw", aligned_gate, raw_gate,
                out_n, 1.0e-5, 1.0e-5);
    compare_f32("IQ2 aligned pair up vs raw", aligned_up, raw_up,
                out_n, 1.0e-5, 1.0e-5);

    ds4_gpu_tensor_free(du_al); ds4_gpu_tensor_free(dg_al);
    ds4_gpu_tensor_free(du_raw); ds4_gpu_tensor_free(dg_raw);
    ds4_gpu_tensor_free(di); ds4_gpu_tensor_free(dx);
    free(aligned_up); free(aligned_gate); free(raw_up); free(raw_gate);
    free(ids); free(x);
}

static void test_iq2_q3_handoff_contract(void *map, uint64_t map_size,
                                         uint64_t gate_off, uint64_t up_off,
                                         uint64_t iq2_bytes,
                                         uint64_t q3_off, uint64_t q3_bytes) {
    const size_t x_n = (size_t)T_IQ2_TOKENS * T_IQ2_IN;
    const size_t pair_n = (size_t)T_IQ2_TOKENS * T_MOE_USED;
    const size_t mid_n = pair_n * T_IQ2_OUT;
    const size_t down_n = pair_n * T_MOE_OUT;
    float *x = malloc(x_n * sizeof(*x));
    int32_t *ids = malloc(pair_n * sizeof(*ids));
    float *weights = malloc(pair_n * sizeof(*weights));
    float *want = malloc(down_n * sizeof(*want));
    float *got = malloc(down_n * sizeof(*got));
    REQUIRE(x && ids && weights && want && got,
            "IQ2/Q3 handoff host arrays");
    for (size_t i = 0; i < x_n; i++) {
        x[i] = 0.35f * sinf(0.009f * (float)(i + 1u)) +
               0.15f * cosf(0.023f * (float)(i + 3u));
    }
    for (size_t p = 0; p < pair_n; p++) {
        ids[p] = (int32_t)((p * 11u + p / T_MOE_USED * 5u) % T_MOE_EXPERT);
        weights[p] = 0.03f + 0.002f * (float)(p % 23u);
    }

    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(x_n * sizeof(*x));
    ds4_gpu_tensor *di = ds4_gpu_tensor_alloc(pair_n * sizeof(*ids));
    ds4_gpu_tensor *dw = ds4_gpu_tensor_alloc(pair_n * sizeof(*weights));
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc(mid_n * sizeof(float));
    ds4_gpu_tensor *du = ds4_gpu_tensor_alloc(mid_n * sizeof(float));
    ds4_gpu_tensor *dm = ds4_gpu_tensor_alloc(mid_n * sizeof(float));
    ds4_gpu_tensor *dy_ref = ds4_gpu_tensor_alloc(down_n * sizeof(float));
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(down_n * sizeof(float));
    REQUIRE(dx && di && dw && dg && du && dm && dy_ref && dy,
            "IQ2/Q3 handoff tensors");
    REQUIRE(ds4_gpu_tensor_write(dx, 0, x, x_n * sizeof(*x)),
            "write IQ2/Q3 activation");
    REQUIRE(ds4_gpu_tensor_write(di, 0, ids, pair_n * sizeof(*ids)),
            "write IQ2/Q3 ids");
    REQUIRE(ds4_gpu_tensor_write(dw, 0, weights, pair_n * sizeof(*weights)),
            "write IQ2/Q3 router weights");
    REQUIRE(ds4_gpu_routed_gate_up_tensor(
                dg, du, dx, di, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes, 16u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_EXPERT,
                T_IQ2_TOKENS, T_MOE_USED),
            "IQ2/Q3 classic gate/up");
    REQUIRE(ds4_gpu_swiglu_weighted_tensor(
                dm, dg, du, dw, T_IQ2_OUT, mid_n),
            "IQ2/Q3 classic weighted SwiGLU");
    REQUIRE(ds4_gpu_routed_matmul_bounded_tensor(
                dy_ref, dm, di, map, map_size, q3_off, q3_bytes, 11u,
                T_IQ2_OUT, T_MOE_OUT, T_MOE_EXPERT,
                (uint32_t)pair_n, 1u, T_IQ2_TOKENS),
            "IQ2/Q3 classic Q3 down");
    REQUIRE(ds4_gpu_tensor_read(dy_ref, 0, want, down_n * sizeof(*want)),
            "read IQ2/Q3 classic down");

    REQUIRE(unsetenv("DS4_CUDA_MOE_IQ2_Q3_HANDOFF") == 0,
            "use default-on IQ2/Q3 handoff policy");
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, T_MOE_EXPERT,
                T_IQ2_TOKENS, T_MOE_USED) == 1,
            "default-unset IQ2/Q3 fused handoff accepted");
    REQUIRE(ds4_cuda_moe_iq2_q3_handoff_launches() > 0,
            "IQ2/Q3 fused handoff launch proof");
    REQUIRE(ds4_gpu_tensor_read(dy, 0, got, down_n * sizeof(*got)),
            "read IQ2/Q3 fused down");
    compare_f32("IQ2/Q3 handoff vs classic", got, want, down_n,
                2.0e-3, 2.0e-2);

    const unsigned long long default_launches =
        ds4_cuda_moe_iq2_q3_handoff_launches();
    REQUIRE(setenv("DS4_CUDA_MOE_IQ2_Q3_HANDOFF", "1", 1) == 0,
            "explicitly enable IQ2/Q3 handoff");
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, T_MOE_EXPERT,
                T_IQ2_TOKENS, T_MOE_USED) == 1,
            "explicit-1 IQ2/Q3 fused handoff accepted");
    REQUIRE(ds4_cuda_moe_iq2_q3_handoff_launches() > default_launches,
            "explicit-1 IQ2/Q3 launch proof");

    /* Decode, sub-threshold batches, and Q4 are pre-launch refusals, so the
     * caller can safely run the established chain. */
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, T_MOE_EXPERT,
                1u, T_MOE_USED) == 0,
            "IQ2/Q3 handoff refuses decode");
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, T_MOE_EXPERT,
                4u, T_MOE_USED) == 0,
            "IQ2/Q3 handoff refuses continuous decode width");
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, T_MOE_EXPERT,
                511u, T_MOE_USED) == 0,
            "IQ2/Q3 handoff refuses 511-token boundary");
    const unsigned long long overflow_launches =
        ds4_cuda_moe_iq2_q3_handoff_launches();
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, (uint32_t)INT_MAX,
                T_IQ2_TOKENS, T_MOE_USED) == 0,
            "IQ2/Q3 handoff refuses INT_MAX experts");
    REQUIRE(ds4_cuda_moe_iq2_q3_handoff_launches() == overflow_launches,
            "overflow refusal leaves launch counter unchanged");
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, 65536u,
                T_IQ2_TOKENS, T_MOE_USED) == 0,
            "IQ2/Q3 handoff refuses CUDA grid-z overflow");
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, 32769u,
                T_IQ2_TOKENS, T_MOE_USED) == 0,
            "IQ2/Q3 handoff refuses signed D2R expert packing overflow");
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, 1048576u, T_MOE_OUT, T_MOE_EXPERT,
                T_IQ2_TOKENS, T_MOE_USED) == 0,
            "IQ2/Q3 handoff refuses gate/up int-span overflow");
    REQUIRE(ds4_cuda_moe_iq2_q3_handoff_launches() == overflow_launches,
            "gate/up overflow refusal leaves launch counter unchanged");
    REQUIRE(!ds4_gpu_swiglu_weighted_q8_d4_emit_test(
                dm, dg, du, dw, di, 0x80000000u, 1u),
            "D4 diagnostic refuses mid_dim above INT_MAX");
    REQUIRE(!ds4_gpu_q3_quantize_ref_test(
                dm, dg, di, T_IQ2_OUT, 0x80000000u),
            "D4 diagnostic refuses assignment count above INT_MAX");
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 12u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, T_MOE_EXPERT,
                T_IQ2_TOKENS, T_MOE_USED) == 0,
            "IQ2/Q3 handoff refuses Q4 down");

    ds4_gpu_tensor_free(dy); ds4_gpu_tensor_free(dy_ref);
    ds4_gpu_tensor_free(dm); ds4_gpu_tensor_free(du); ds4_gpu_tensor_free(dg);
    ds4_gpu_tensor_free(dw); ds4_gpu_tensor_free(di); ds4_gpu_tensor_free(dx);
    free(got); free(want);
    free(weights); free(ids); free(x);
}

static void fill_sentinel(unsigned char *dst, size_t bytes, uint8_t seed) {
    for (size_t i = 0; i < bytes; i++) {
        dst[i] = (uint8_t)(seed + 29u * (uint8_t)i);
    }
}

static void test_iq2_q3_handoff_prelaunch_refusal(
        void *map, uint64_t map_size, uint64_t gate_off, uint64_t up_off,
        uint64_t iq2_bytes, uint64_t q3_off, uint64_t q3_bytes,
        const char *case_name) {
    const size_t x_n = (size_t)T_IQ2_TOKENS * T_IQ2_IN;
    const size_t pair_n = (size_t)T_IQ2_TOKENS * T_MOE_USED;
    const size_t mid_bytes = pair_n * T_IQ2_OUT * sizeof(float);
    const size_t down_bytes = pair_n * T_MOE_OUT * sizeof(float);
    float *x = calloc(x_n, sizeof(*x));
    int32_t *ids = calloc(pair_n, sizeof(*ids));
    float *weights = calloc(pair_n, sizeof(*weights));
    unsigned char *before = malloc(mid_bytes);
    unsigned char *after = malloc(mid_bytes);
    REQUIRE(x && ids && weights && before && after,
            "prelaunch refusal host arrays");
    for (size_t p = 0; p < pair_n; p++) {
        ids[p] = (int32_t)(p % T_MOE_EXPERT);
        weights[p] = 0.125f;
    }

    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(x_n * sizeof(*x));
    ds4_gpu_tensor *di = ds4_gpu_tensor_alloc(pair_n * sizeof(*ids));
    ds4_gpu_tensor *dw = ds4_gpu_tensor_alloc(pair_n * sizeof(*weights));
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *du = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *dm = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(down_bytes);
    REQUIRE(dx && di && dw && dg && du && dm && dy,
            "prelaunch refusal tensors");
    REQUIRE(ds4_gpu_tensor_write(dx, 0, x, x_n * sizeof(*x)),
            "write prelaunch-refusal activation");
    REQUIRE(ds4_gpu_tensor_write(di, 0, ids, pair_n * sizeof(*ids)),
            "write prelaunch-refusal ids");
    REQUIRE(ds4_gpu_tensor_write(dw, 0, weights, pair_n * sizeof(*weights)),
            "write prelaunch-refusal router weights");

#define CHECK_REFUSAL_SENTINEL(tensor_, bytes_, seed_, label_) do {          \
        fill_sentinel(before, (bytes_), (seed_));                            \
        REQUIRE(ds4_gpu_tensor_write((tensor_), 0, before, (bytes_)),        \
                "write " label_ " sentinel");                             \
    } while (0)
    CHECK_REFUSAL_SENTINEL(dg, mid_bytes, 0x11u, "gate");
    CHECK_REFUSAL_SENTINEL(du, mid_bytes, 0x37u, "up");
    CHECK_REFUSAL_SENTINEL(dm, mid_bytes, 0x59u, "Q8 scratch");
    CHECK_REFUSAL_SENTINEL(dy, down_bytes, 0x7bu, "down");
#undef CHECK_REFUSAL_SENTINEL

    const unsigned long long launches_before =
        ds4_cuda_moe_iq2_q3_handoff_launches();
    REQUIRE(ds4_gpu_routed_iq2_q3_handoff_tensor(
                dy, dg, du, dm, dx, di, dw, map, map_size,
                gate_off, iq2_bytes, up_off, iq2_bytes,
                q3_off, q3_bytes, 16u, 11u,
                T_IQ2_IN, T_IQ2_OUT, T_MOE_OUT, T_MOE_EXPERT,
                T_IQ2_TOKENS, T_MOE_USED) == 0,
            "IQ2/Q3 handoff refused before launch");
    REQUIRE(ds4_cuda_moe_iq2_q3_handoff_launches() == launches_before,
            "prelaunch refusal does not increment launch counter");

#define REQUIRE_REFUSAL_SENTINEL(tensor_, bytes_, seed_, label_) do {        \
        fill_sentinel(before, (bytes_), (seed_));                            \
        REQUIRE(ds4_gpu_tensor_read((tensor_), 0, after, (bytes_)),          \
                "read " label_ " sentinel");                              \
        REQUIRE(memcmp(before, after, (bytes_)) == 0,                        \
                label_ " unchanged after refusal");                        \
    } while (0)
    REQUIRE_REFUSAL_SENTINEL(dg, mid_bytes, 0x11u, "gate");
    REQUIRE_REFUSAL_SENTINEL(du, mid_bytes, 0x37u, "up");
    REQUIRE_REFUSAL_SENTINEL(dm, mid_bytes, 0x59u, "Q8 scratch");
    REQUIRE_REFUSAL_SENTINEL(dy, down_bytes, 0x7bu, "down");
#undef REQUIRE_REFUSAL_SENTINEL

    printf("%-38s ok\n", case_name);

    ds4_gpu_tensor_free(dy); ds4_gpu_tensor_free(dm);
    ds4_gpu_tensor_free(du); ds4_gpu_tensor_free(dg);
    ds4_gpu_tensor_free(dw); ds4_gpu_tensor_free(di); ds4_gpu_tensor_free(dx);
    free(after); free(before); free(weights); free(ids); free(x);
}

static void test_iq2_q3_xmax64_subprocess(void) {
    const pid_t pid = fork();
    REQUIRE(pid >= 0, "fork X_MAX=64 refusal subprocess");
    if (pid == 0) {
        REQUIRE(setenv("DS4_CUDA_MMQ_X_MAX", "64", 1) == 0,
                "set X_MAX=64 in refusal subprocess");
        REQUIRE(setenv("DS4_CUDA_MOE_IQ2_Q3_HANDOFF", "1", 1) == 0,
                "enable handoff in refusal subprocess");
        execl("/proc/self/exe", "test_model_family_kernels",
              "--iq2-q3-xmax64-refusal", (char *)NULL);
        _exit(127);
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid,
            "wait X_MAX=64 refusal subprocess");
    REQUIRE(WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "X_MAX=64 refusal subprocess passed");
}

static void test_iq2_q3_disabled_policy_subprocess(void) {
    const pid_t pid = fork();
    REQUIRE(pid >= 0, "fork disabled-policy refusal subprocess");
    if (pid == 0) {
        REQUIRE(unsetenv("DS4_CUDA_MMQ_X_MAX") == 0,
                "unset X_MAX in disabled-policy subprocess");
        REQUIRE(setenv("DS4_CUDA_MOE_IQ2_Q3_HANDOFF", "0", 1) == 0,
                "disable handoff in policy subprocess");
        execl("/proc/self/exe", "test_model_family_kernels",
              "--iq2-q3-disabled-policy-refusal", (char *)NULL);
        _exit(127);
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid,
            "wait disabled-policy refusal subprocess");
    REQUIRE(WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "disabled-policy refusal subprocess passed");
}

int main(int argc, char **argv) {
    const int xmax64_refusal_only =
        argc == 2 && strcmp(argv[1], "--iq2-q3-xmax64-refusal") == 0;
    const int disabled_policy_refusal_only =
        argc == 2 &&
        strcmp(argv[1], "--iq2-q3-disabled-policy-refusal") == 0;
    const int refusal_only =
        xmax64_refusal_only || disabled_policy_refusal_only;
    REQUIRE(argc == 1 || refusal_only,
            "recognized model-family test argument");
    if (!refusal_only) {
        test_iq2_q3_xmax64_subprocess();
        test_iq2_q3_disabled_policy_subprocess();
    }
    REQUIRE(ds4_gpu_init(), "CUDA init");
    const uint64_t embed_off = 4096u;
    const uint64_t embed_bytes = (uint64_t)T_VOCAB * (T_EMBD / 32u) * 34u;
    const uint64_t bias_off = (embed_off + embed_bytes + 4095u) & ~4095ull;
    const uint64_t norm_off = bias_off + 2048u;
    const uint64_t q3_off = bias_off + 4096u;
    const uint64_t q3_bytes =
        (uint64_t)T_MOE_EXPERT * T_MOE_OUT * sizeof(test_block_q3_k);
    const uint64_t iq2_gate_off = (q3_off + q3_bytes + 4095u) & ~4095ull;
    const uint64_t iq2_blocks =
        (uint64_t)T_MOE_EXPERT * T_IQ2_OUT * (T_IQ2_IN / 256u);
    const uint64_t iq2_bytes = iq2_blocks * 66u;
    const uint64_t iq2_up_off =
        (iq2_gate_off + iq2_bytes + 4095u) & ~4095ull;
    const uint64_t handoff_q3_off =
        (iq2_up_off + iq2_bytes + 4095u) & ~4095ull;
    const uint64_t handoff_q3_blocks =
        (uint64_t)T_MOE_EXPERT * T_MOE_OUT * (T_IQ2_OUT / 256u);
    const uint64_t handoff_q3_bytes =
        handoff_q3_blocks * sizeof(test_block_q3_k);
    const uint64_t map_size =
        (handoff_q3_off + handoff_q3_bytes + 4095u) & ~4095ull;
    void *map = NULL;
    REQUIRE(posix_memalign(&map, 4096u, (size_t)map_size) == 0, "model map alloc");
    memset(map, 0, (size_t)map_size);
    fill_q8_embedding((unsigned char *)map + embed_off);
    fill_q3_weights((test_block_q3_k *)((unsigned char *)map + q3_off),
                    (size_t)T_MOE_EXPERT * T_MOE_OUT, 0x51a7c0deu);
    fill_q3_weights(
        (test_block_q3_k *)((unsigned char *)map + handoff_q3_off),
        (size_t)handoff_q3_blocks, 0x7e57c0deu);
    fill_iq2_weights((unsigned char *)map + iq2_gate_off,
                     iq2_blocks, 0x13579bdfu);
    fill_iq2_weights((unsigned char *)map + iq2_up_off,
                     iq2_blocks, 0x2468ace0u);
    REQUIRE(ds4_gpu_set_model_map(map, map_size), "model map registration");

    if (refusal_only) {
        build_iq2_artifacts(map, map_size, iq2_gate_off, iq2_up_off,
                            iq2_bytes);
        test_iq2_q3_handoff_prelaunch_refusal(
            map, map_size, iq2_gate_off, iq2_up_off, iq2_bytes,
            handoff_q3_off, handoff_q3_bytes,
            xmax64_refusal_only ? "X_MAX=64 prelaunch refusal"
                                : "env=0 prelaunch refusal");
        if (disabled_policy_refusal_only) {
            REQUIRE(setenv("DS4_CUDA_MOE_IQ2_Q3_HANDOFF", "invalid", 1) == 0,
                    "set invalid handoff policy value");
            test_iq2_q3_handoff_prelaunch_refusal(
                map, map_size, iq2_gate_off, iq2_up_off, iq2_bytes,
                handoff_q3_off, handoff_q3_bytes,
                "invalid-env prelaunch refusal");
        }
        ds4_gpu_unregister_model_map(map);
        free(map);
        ds4_gpu_cleanup();
        puts(xmax64_refusal_only
                 ? "IQ2/Q3 X_MAX=64 prelaunch refusal passed"
                 : "IQ2/Q3 disabled policy refusals passed");
        return failures ? 1 : 0;
    }

    puts("== generic model-family CUDA primitives ==");
    test_embedding(map, map_size, embed_off);
    test_router(map, map_size, bias_off);
    test_weighted_swiglu();
    test_weighted_swiglu_q8_d4_bytes();
    test_q3_worklist_index_bounds();
    test_rms_norm_q8_producer(map, map_size, norm_off);
    test_q3_routed_matmul(map, map_size, q3_off, q3_bytes);
    test_iq2_aligned_pair(map, map_size, iq2_gate_off, iq2_up_off,
                          iq2_bytes);
    test_iq2_q3_handoff_contract(map, map_size,
                                 iq2_gate_off, iq2_up_off, iq2_bytes,
                                 handoff_q3_off, handoff_q3_bytes);

    ds4_gpu_unregister_model_map(map);
    free(map);
    ds4_gpu_cleanup();
    puts(failures ? "model-family primitive checks FAILED"
                  : "all model-family primitive checks passed");
    return failures ? 1 : 0;
}
