/* Solar Open 2 compressed GQA KV checks.
 *
 * The test keeps the real head width (128), exercises GQA head grouping,
 * compares FP8/FP4/hybrid attention with the BF16 oracle, and checks split-KV
 * decode against the one-block path over the same compressed cache.
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    T_HEAD = 8,
    T_HEAD_KV = 2,
    T_DIM = 128,
    T_TOKENS = 257,
    T_CAP = 320,
    T_CHUNK = 64,
};

static int failures = 0;

#define CHECK(condition, message)                                             \
    do {                                                                      \
        if (!(condition)) {                                                   \
            fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);    \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16u;
    uint32_t exp = (h >> 10u) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1u;
                shift++;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(127 - 15 - shift) << 23u) |
                   (mant << 13u);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13u);
    } else {
        bits = sign | ((exp + 112u) << 23u) | (mant << 13u);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float e4m3_value(uint8_t code) {
    const uint8_t mag = code & 0x7fu;
    const int exp = (mag >> 3u) & 15;
    const int mant = mag & 7;
    float v = exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
    return (code & 0x80u) ? -v : v;
}

static float e2m1_value(uint8_t code) {
    static const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
    const float v = table[code & 7u];
    return (code & 8u) ? -v : v;
}

static float cache_value(const uint8_t *row, ds4_solar_kv_format format,
                         uint32_t head, uint32_t dim, int value) {
    const uint64_t kv_dim = (uint64_t)T_HEAD_KV * T_DIM;
    const uint64_t elem = (uint64_t)head * T_DIM + dim;
    if (format == DS4_SOLAR_KV_BF16) {
        const uint16_t *p = (const uint16_t *)row;
        return f16_to_f32(p[(value ? kv_dim : 0u) + elem]);
    }
    const uint64_t k_bytes =
        format == DS4_SOLAR_KV_FP4 ? kv_dim / 2u : kv_dim;
    const uint64_t v_bytes =
        format == DS4_SOLAR_KV_FP8 ? kv_dim : kv_dim / 2u;
    const uint16_t *scales =
        (const uint16_t *)(row + k_bytes + v_bytes);
    const float scale = f16_to_f32(scales[(value ? T_HEAD_KV : 0u) + head]);
    const uint8_t *data = value ? row + k_bytes : row;
    const int fp8 = value ? format == DS4_SOLAR_KV_FP8
                          : format != DS4_SOLAR_KV_FP4;
    if (fp8) return e4m3_value(data[elem]) * scale;
    const uint8_t packed = data[elem >> 1u];
    return e2m1_value((elem & 1u) ? packed >> 4u : packed & 0x0fu) * scale;
}

static const char *format_name(ds4_solar_kv_format format) {
    switch (format) {
    case DS4_SOLAR_KV_BF16: return "BF16";
    case DS4_SOLAR_KV_FP8: return "FP8";
    case DS4_SOLAR_KV_FP4: return "FP4";
    case DS4_SOLAR_KV_KFP8_VFP4: return "K-FP8/V-FP4";
    default: return "invalid";
    }
}

static uint32_t repeat_count_from_env(const char *name) {
    const char *value = getenv(name);
    if (!value || !value[0]) return 1u;
    char *end = NULL;
    const unsigned long count = strtoul(value, &end, 10);
    if (end == value || *end != '\0' || count == 0u || count > 10000u) {
        return 0u;
    }
    return (uint32_t)count;
}

static void compare_vector(const char *label, const float *got,
                           const float *want, size_t n, double rel_limit,
                           double cosine_limit) {
    double err2 = 0.0, ref2 = 0.0, got2 = 0.0, dot = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double g = got[i], r = want[i], d = g - r;
        err2 += d * d;
        ref2 += r * r;
        got2 += g * g;
        dot += g * r;
        if (fabs(d) > max_abs) max_abs = fabs(d);
    }
    const double rel = ref2 > 0.0 ? sqrt(err2 / ref2) : sqrt(err2);
    const double cosine = ref2 > 0.0 && got2 > 0.0
        ? dot / sqrt(ref2 * got2) : 1.0;
    const double cosine_error = 1.0 - cosine;
    const int ok = rel <= rel_limit && cosine_error <= cosine_limit;
    if (!ok) failures++;
    printf("%-34s rel_rms=%.4e 1-cos=%.4e max_abs=%.4e %s\n",
           label, rel, cosine_error, max_abs, ok ? "ok" : "FAIL");
}

static void make_fixture(float *q, float *k, float *v) {
    for (uint32_t t = 0; t < T_TOKENS; t++) {
        for (uint32_t h = 0; h < T_HEAD; h++) {
            for (uint32_t d = 0; d < T_DIM; d++) {
                const size_t i = ((size_t)t * T_HEAD + h) * T_DIM + d;
                const float x = (float)(1u + d + 17u * h + 3u * t);
                q[i] = 0.55f * sinf(0.013f * x) +
                       0.17f * cosf(0.021f * x + 0.1f * (float)h);
            }
        }
        for (uint32_t h = 0; h < T_HEAD_KV; h++) {
            for (uint32_t d = 0; d < T_DIM; d++) {
                const size_t i = ((size_t)t * T_HEAD_KV + h) * T_DIM + d;
                const float x = (float)(1u + d + 29u * h + 5u * t);
                k[i] = 0.62f * sinf(0.017f * x) +
                       0.11f * cosf(0.031f * x);
                v[i] = 0.48f * cosf(0.019f * x + 0.2f) -
                       0.16f * sinf(0.027f * x);
                if (((t * 131u + h * 17u + d) % 997u) == 0u) k[i] *= 3.5f;
                if (((t * 83u + h * 23u + d) % 911u) == 0u) v[i] *= -3.0f;
            }
        }
    }
}

int main(void) {
    setenv("DS4_EXAONE_NO_PREFILL_HMMA", "1", 1);
    const uint32_t prefill_repeats =
        repeat_count_from_env("DS4_SOLAR_KV_PREFILL_REPEATS");
    const uint32_t decode_repeats =
        repeat_count_from_env("DS4_SOLAR_KV_DECODE_REPEATS");
    CHECK(prefill_repeats != 0u,
          "DS4_SOLAR_KV_PREFILL_REPEATS must be in [1, 10000]");
    CHECK(decode_repeats != 0u,
          "DS4_SOLAR_KV_DECODE_REPEATS must be in [1, 10000]");
    CHECK(ds4_gpu_init(), "CUDA init");

    const size_t q_count = (size_t)T_TOKENS * T_HEAD * T_DIM;
    const size_t kv_count = (size_t)T_TOKENS * T_HEAD_KV * T_DIM;
    const size_t row_count = (size_t)T_HEAD * T_DIM;
    float *q = malloc(q_count * sizeof(float));
    float *k = malloc(kv_count * sizeof(float));
    float *v = malloc(kv_count * sizeof(float));
    float *baseline_prefill = malloc(q_count * sizeof(float));
    float *baseline_decode = malloc(row_count * sizeof(float));
    float *got = malloc(q_count * sizeof(float));
    float *got_decode = malloc(row_count * sizeof(float));
    float *got_split = malloc(row_count * sizeof(float));
    CHECK(q && k && v && baseline_prefill && baseline_decode && got &&
          got_decode && got_split, "host allocation");
    make_fixture(q, k, v);

    ds4_gpu_tensor *dq = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *dk = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
    ds4_gpu_tensor *dv = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *decode = ds4_gpu_tensor_alloc(row_count * sizeof(float));
    ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(row_count * sizeof(float));
    const uint32_t parts_n = (T_TOKENS + T_CHUNK - 1u) / T_CHUNK;
    ds4_gpu_tensor *partials = ds4_gpu_tensor_alloc(
        (uint64_t)T_HEAD * parts_n * (T_DIM + 2u) * sizeof(float));
    CHECK(dq && dk && dv && out && decode && split && partials,
          "device allocation");
    CHECK(ds4_gpu_tensor_write(dq, 0, q, q_count * sizeof(float)), "write q");
    CHECK(ds4_gpu_tensor_write(dk, 0, k, kv_count * sizeof(float)), "write k");
    CHECK(ds4_gpu_tensor_write(dv, 0, v, kv_count * sizeof(float)), "write v");
    ds4_gpu_tensor *last_q = ds4_gpu_tensor_view(
        dq, (uint64_t)(T_TOKENS - 1u) * row_count * sizeof(float),
        (uint64_t)row_count * sizeof(float));
    CHECK(last_q, "last query view");

    const uint64_t bf16_row = ds4_gpu_solar_kv_row_bytes(
        DS4_SOLAR_KV_BF16, T_HEAD_KV, T_DIM);
    ds4_gpu_tensor *bf16 = ds4_gpu_tensor_alloc((uint64_t)T_CAP * bf16_row);
    CHECK(bf16, "BF16 cache allocation");
    CHECK(ds4_gpu_solar_kv_store_tensor(
              bf16, dk, dv, T_HEAD_KV, T_DIM, T_TOKENS, 0u, T_CAP,
              DS4_SOLAR_KV_BF16),
          "BF16 cache store");
    CHECK(ds4_gpu_solar_attention_prefill_tensor(
              out, dq, bf16, T_TOKENS, 0u, T_HEAD, T_HEAD_KV, T_DIM, T_CAP,
              0u, DS4_SOLAR_KV_BF16),
          "BF16 prefill");
    CHECK(ds4_gpu_tensor_read(
              out, 0, baseline_prefill, q_count * sizeof(float)),
          "read BF16 prefill");
    CHECK(ds4_gpu_solar_attention_decode_tensor(
              decode, last_q, bf16, T_HEAD, T_HEAD_KV, T_DIM, T_CAP,
              T_TOKENS - 1u, 0u, DS4_SOLAR_KV_BF16),
          "BF16 decode");
    CHECK(ds4_gpu_tensor_read(
              decode, 0, baseline_decode, row_count * sizeof(float)),
          "read BF16 decode");

    puts("== Solar Open 2 compressed GQA KV ==");
    if (prefill_repeats > 1u) {
        printf("compressed prefill repeats=%u\n", prefill_repeats);
    }
    if (decode_repeats > 1u) {
        printf("compressed decode repeats=%u\n", decode_repeats);
    }
    const ds4_solar_kv_format formats[] = {
        DS4_SOLAR_KV_FP8,
        DS4_SOLAR_KV_FP4,
        DS4_SOLAR_KV_KFP8_VFP4,
    };
    for (size_t fi = 0; fi < sizeof(formats) / sizeof(formats[0]); fi++) {
        const ds4_solar_kv_format format = formats[fi];
        const uint64_t row_bytes =
            ds4_gpu_solar_kv_row_bytes(format, T_HEAD_KV, T_DIM);
        ds4_gpu_tensor *cache =
            ds4_gpu_tensor_alloc((uint64_t)T_CAP * row_bytes);
        CHECK(cache, "compressed cache allocation");
        CHECK(ds4_gpu_solar_kv_store_tensor(
                  cache, dk, dv, T_HEAD_KV, T_DIM, T_TOKENS, 0u, T_CAP,
                  format),
              "compressed cache store");

        uint8_t *cache_host = malloc((size_t)T_CAP * row_bytes);
        CHECK(cache_host, "compressed host cache allocation");
        CHECK(ds4_gpu_tensor_read(
                  cache, 0, cache_host, (uint64_t)T_CAP * row_bytes),
              "read compressed cache");
        double ke2 = 0.0, kr2 = 0.0, ve2 = 0.0, vr2 = 0.0;
        for (uint32_t t = 0; t < T_TOKENS; t++) {
            const uint8_t *row = cache_host + (uint64_t)t * row_bytes;
            for (uint32_t h = 0; h < T_HEAD_KV; h++) {
                for (uint32_t d = 0; d < T_DIM; d++) {
                    const size_t i = ((size_t)t * T_HEAD_KV + h) * T_DIM + d;
                    const double kd = cache_value(row, format, h, d, 0) - k[i];
                    const double vd = cache_value(row, format, h, d, 1) - v[i];
                    ke2 += kd * kd;
                    kr2 += (double)k[i] * k[i];
                    ve2 += vd * vd;
                    vr2 += (double)v[i] * v[i];
                }
            }
        }
        const double krel = sqrt(ke2 / kr2);
        const double vrel = sqrt(ve2 / vr2);
        const double klim = format == DS4_SOLAR_KV_FP4 ? 0.25 : 0.05;
        const double vlim = format == DS4_SOLAR_KV_FP8 ? 0.05 : 0.25;
        const int cache_ok = krel <= klim && vrel <= vlim;
        if (!cache_ok) failures++;
        printf("%-34s K rel=%.4e V rel=%.4e row=%llu B %s\n",
               format_name(format), krel, vrel,
               (unsigned long long)row_bytes, cache_ok ? "ok" : "FAIL");

        for (uint32_t repeat = 0; repeat < prefill_repeats; repeat++) {
            CHECK(ds4_gpu_solar_attention_prefill_tensor(
                      out, dq, cache, T_TOKENS, 0u, T_HEAD, T_HEAD_KV, T_DIM,
                      T_CAP, 0u, format),
                  "compressed prefill");
        }
        CHECK(ds4_gpu_tensor_read(out, 0, got, q_count * sizeof(float)),
              "read compressed prefill");
        const double attn_limit = format == DS4_SOLAR_KV_FP8 ? 0.06 : 0.25;
        char label[64];
        snprintf(label, sizeof(label), "%s prefill vs BF16", format_name(format));
        compare_vector(label, got, baseline_prefill, q_count, attn_limit, 0.04);

        for (uint32_t repeat = 0; repeat < decode_repeats; repeat++) {
            CHECK(ds4_gpu_solar_attention_decode_tensor(
                      decode, last_q, cache, T_HEAD, T_HEAD_KV, T_DIM, T_CAP,
                      T_TOKENS - 1u, 0u, format),
                  "compressed decode");
        }
        CHECK(ds4_gpu_tensor_read(
                  decode, 0, got_decode, row_count * sizeof(float)),
              "read compressed decode");
        snprintf(label, sizeof(label), "%s decode vs BF16", format_name(format));
        compare_vector(label, got_decode, baseline_decode, row_count,
                       attn_limit, 0.04);

        CHECK(ds4_gpu_solar_attention_decode_split_tensor(
                  split, last_q, cache, partials, T_HEAD, T_HEAD_KV, T_DIM,
                  T_CAP, T_TOKENS - 1u, 0u, T_CHUNK, format),
              "compressed split decode");
        CHECK(ds4_gpu_tensor_read(
                  split, 0, got_split, row_count * sizeof(float)),
              "read compressed split decode");
        snprintf(label, sizeof(label), "%s split vs direct", format_name(format));
        compare_vector(label, got_split, got_decode, row_count, 2.0e-5, 2.0e-5);

        free(cache_host);
        ds4_gpu_tensor_free(cache);
    }

    const uint64_t ctx = 1048576u;
    puts("== 1M / 12 GQA-layer KV projection ==");
    for (int f = DS4_SOLAR_KV_BF16; f <= DS4_SOLAR_KV_KFP8_VFP4; f++) {
        const uint64_t row = ds4_gpu_solar_kv_row_bytes(
            (ds4_solar_kv_format)f, 8u, 128u);
        const uint64_t bytes = row * ctx * 12u;
        printf("%-14s row=%4llu B total=%.3f GiB\n",
               format_name((ds4_solar_kv_format)f),
               (unsigned long long)row,
               (double)bytes / 1073741824.0);
    }

    ds4_gpu_tensor_free(bf16);
    ds4_gpu_tensor_free(last_q);
    ds4_gpu_tensor_free(partials);
    ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(decode);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(dv);
    ds4_gpu_tensor_free(dk);
    ds4_gpu_tensor_free(dq);
    free(got_split);
    free(got_decode);
    free(got);
    free(baseline_decode);
    free(baseline_prefill);
    free(v);
    free(k);
    free(q);
    ds4_gpu_cleanup();
    puts(failures ? "Solar compressed KV checks FAILED"
                  : "all Solar compressed KV checks passed");
    return failures ? 1 : 0;
}
