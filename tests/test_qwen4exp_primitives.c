/* CUDA parity for the Qwen3.8-Flash-Next zero-centred RMSNorm and
 * hyper-connection elementwise stages.  The dimensions exercise the exact
 * four-stream, hidden-size-2560 production layout; CPU references are kept
 * independent of the CUDA implementation. */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    QWEN_ROWS = 3,
    QWEN_HIDDEN = 2560,
    QWEN_HC = 4,
    QWEN_WIDTH = QWEN_HIDDEN * QWEN_HC,
    QWEN_LOWRANK = 320,
};

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static void write_f32(ds4_gpu_tensor *tensor, const float *src, uint64_t n,
                      const char *what) {
    REQUIRE(ds4_gpu_tensor_write(tensor, 0, src, n * sizeof(*src)), what);
}

static void read_f32(const ds4_gpu_tensor *tensor, float *dst, uint64_t n,
                     const char *what) {
    REQUIRE(ds4_gpu_tensor_read(tensor, 0, dst, n * sizeof(*dst)), what);
}

static void compare_f32(const char *name, const float *got, const float *want,
                        uint64_t n, float atol, float rtol) {
    double worst_ratio = 0.0;
    float worst_abs = 0.0f;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        const float abs_err = fabsf(got[i] - want[i]);
        const float limit = atol + rtol * fabsf(want[i]);
        const double ratio = limit > 0.0f ? (double)abs_err / limit
                                          : (double)abs_err;
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
            worst_abs = abs_err;
            worst_i = i;
        }
    }
    if (worst_ratio > 1.0) {
        fprintf(stderr,
                "FAIL: %s at %llu: got %.9g want %.9g abs %.9g (%.3fx limit)\n",
                name, (unsigned long long)worst_i, got[worst_i], want[worst_i],
                worst_abs, worst_ratio);
        exit(1);
    }
    printf("%-43s pass (worst abs %.3g, %.3fx limit)\n",
           name, worst_abs, worst_ratio);
}

static float sigmoid_ref(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void test_group_rms_norm(void *model_map, uint64_t model_size,
                                uint64_t weight_offset) {
    const uint64_t count = (uint64_t)QWEN_ROWS * QWEN_WIDTH;
    float *input = (float *)malloc(count * sizeof(*input));
    float *want = (float *)malloc(count * sizeof(*want));
    float *got = (float *)malloc(count * sizeof(*got));
    float *weight = (float *)((unsigned char *)model_map + weight_offset);
    REQUIRE(input && want && got, "group RMSNorm host allocation");

    for (uint32_t i = 0; i < QWEN_WIDTH; i++)
        weight[i] = 0.17f * sinf((float)(i + 3u) * 0.013f) - 0.04f;
    for (uint64_t i = 0; i < count; i++)
        input[i] = 1.7f * sinf((float)(i + 1u) * 0.0037f) +
                   0.35f * cosf((float)(i + 11u) * 0.017f) +
                   (float)((int)(i % 19u) - 9) * 0.003f;

    for (uint32_t row = 0; row < QWEN_ROWS; row++) {
        for (uint32_t hc = 0; hc < QWEN_HC; hc++) {
            const uint64_t base = (uint64_t)row * QWEN_WIDTH +
                                  (uint64_t)hc * QWEN_HIDDEN;
            double sum = 0.0;
            for (uint32_t d = 0; d < QWEN_HIDDEN; d++) {
                const double v = input[base + d];
                sum += v * v;
            }
            const float scale = 1.0f /
                sqrtf((float)(sum / (double)QWEN_HIDDEN) + 1.0e-6f);
            for (uint32_t d = 0; d < QWEN_HIDDEN; d++)
                want[base + d] = input[base + d] * scale *
                                 (1.0f + weight[hc * QWEN_HIDDEN + d]);
        }
    }

    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(count * sizeof(float));
    REQUIRE(dx && dy, "group RMSNorm GPU allocation");
    write_f32(dx, input, count, "group RMSNorm input upload");
    REQUIRE(ds4_gpu_qwen4exp_group_rms_norm_rows_tensor(
                dy, dx, model_map, model_size, weight_offset,
                QWEN_WIDTH, QWEN_HIDDEN, QWEN_ROWS, 1.0e-6f),
            "Qwen grouped RMSNorm launch");
    read_f32(dy, got, count, "group RMSNorm output download");
    compare_f32("Qwen zero-centred grouped RMSNorm", got, want, count,
                8.0e-6f, 8.0e-6f);

    /* In-place is required by the eventual graph allocator. */
    write_f32(dx, input, count, "group RMSNorm in-place re-upload");
    REQUIRE(ds4_gpu_qwen4exp_group_rms_norm_rows_tensor(
                dx, dx, model_map, model_size, weight_offset,
                QWEN_WIDTH, QWEN_HIDDEN, QWEN_ROWS, 1.0e-6f),
            "Qwen grouped RMSNorm in-place launch");
    read_f32(dx, got, count, "group RMSNorm in-place download");
    compare_f32("Qwen grouped RMSNorm in-place", got, want, count,
                8.0e-6f, 8.0e-6f);

    REQUIRE(!ds4_gpu_qwen4exp_group_rms_norm_rows_tensor(
                dy, dx, model_map, model_size, weight_offset,
                QWEN_WIDTH, 0, QWEN_ROWS, 1.0e-6f),
            "group RMSNorm rejects zero group");
    REQUIRE(!ds4_gpu_qwen4exp_group_rms_norm_rows_tensor(
                dy, dx, model_map, model_size, weight_offset,
                QWEN_WIDTH - 1u, QWEN_HIDDEN, QWEN_ROWS, 1.0e-6f),
            "group RMSNorm rejects non-divisible width");

    ds4_gpu_tensor_free(dy);
    ds4_gpu_tensor_free(dx);
    free(got);
    free(want);
    free(input);
}

static void test_hyper_connection(void) {
    const uint64_t low_count = (uint64_t)QWEN_ROWS * QWEN_LOWRANK;
    const uint64_t hc_count = (uint64_t)QWEN_ROWS * QWEN_WIDTH;
    const uint64_t mixed_count = (uint64_t)QWEN_ROWS * QWEN_HIDDEN;
    const uint64_t inject_count = (uint64_t)QWEN_ROWS * QWEN_HC;
    float *down = (float *)malloc(low_count * sizeof(*down));
    float *down_want = (float *)malloc(low_count * sizeof(*down_want));
    float *down_got = (float *)malloc(low_count * sizeof(*down_got));
    float *normed = (float *)malloc(hc_count * sizeof(*normed));
    float *mix_logits = (float *)malloc(hc_count * sizeof(*mix_logits));
    float *hyper = (float *)malloc(hc_count * sizeof(*hyper));
    float *residual_want = (float *)malloc(hc_count * sizeof(*residual_want));
    float *residual_got = (float *)malloc(hc_count * sizeof(*residual_got));
    float *mixed_want = (float *)malloc(mixed_count * sizeof(*mixed_want));
    float *mixed_got = (float *)malloc(mixed_count * sizeof(*mixed_got));
    float *block = (float *)malloc(mixed_count * sizeof(*block));
    float *inject_logits = (float *)malloc(inject_count * sizeof(*inject_logits));
    float *inject_want = (float *)malloc(inject_count * sizeof(*inject_want));
    float *inject_got = (float *)malloc(inject_count * sizeof(*inject_got));
    REQUIRE(down && down_want && down_got && normed && mix_logits && hyper &&
            residual_want && residual_got && mixed_want && mixed_got && block &&
            inject_logits && inject_want && inject_got,
            "hyper-connection host allocation");

    for (uint64_t i = 0; i < low_count; i++) {
        down[i] = (float)((int)(i % 73u) - 36) * 0.19f;
        const float v = down[i] / (float)QWEN_HC;
        down_want[i] = v * sigmoid_ref(v);
    }
    for (uint64_t i = 0; i < hc_count; i++) {
        normed[i] = 0.9f * sinf((float)(i + 5u) * 0.007f) +
                    0.2f * cosf((float)(i + 9u) * 0.021f);
        mix_logits[i] = (float)((int)(i % 97u) - 48) * 0.071f;
        hyper[i] = 0.7f * cosf((float)(i + 2u) * 0.009f);
    }
    for (uint64_t i = 0; i < mixed_count; i++)
        block[i] = 0.55f * sinf((float)(i + 13u) * 0.012f);
    for (uint64_t i = 0; i < inject_count; i++) {
        inject_logits[i] = (float)((int)i - 6) * 0.83f;
        inject_want[i] = 2.0f * sigmoid_ref(
            inject_logits[i] / (float)QWEN_HC);
    }

    for (uint32_t row = 0; row < QWEN_ROWS; row++) {
        for (uint32_t d = 0; d < QWEN_HIDDEN; d++) {
            float sum = 0.0f;
            for (uint32_t hc = 0; hc < QWEN_HC; hc++) {
                const uint64_t at = (uint64_t)row * QWEN_WIDTH +
                                    (uint64_t)hc * QWEN_HIDDEN + d;
                sum += sigmoid_ref(mix_logits[at]) * normed[at];
            }
            mixed_want[(uint64_t)row * QWEN_HIDDEN + d] =
                sum / (float)QWEN_HC;
        }
    }
    for (uint32_t row = 0; row < QWEN_ROWS; row++) {
        for (uint32_t hc = 0; hc < QWEN_HC; hc++) {
            for (uint32_t d = 0; d < QWEN_HIDDEN; d++) {
                const uint64_t at = (uint64_t)row * QWEN_WIDTH +
                                    (uint64_t)hc * QWEN_HIDDEN + d;
                residual_want[at] = hyper[at] +
                    block[(uint64_t)row * QWEN_HIDDEN + d] *
                    inject_want[(uint64_t)row * QWEN_HC + hc];
            }
        }
    }

    ds4_gpu_tensor *d_down = ds4_gpu_tensor_alloc(low_count * sizeof(float));
    ds4_gpu_tensor *d_down_out = ds4_gpu_tensor_alloc(low_count * sizeof(float));
    ds4_gpu_tensor *d_normed = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
    ds4_gpu_tensor *d_mix_logits = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
    ds4_gpu_tensor *d_hyper = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
    ds4_gpu_tensor *d_residual = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
    ds4_gpu_tensor *d_mixed = ds4_gpu_tensor_alloc(mixed_count * sizeof(float));
    ds4_gpu_tensor *d_block = ds4_gpu_tensor_alloc(mixed_count * sizeof(float));
    ds4_gpu_tensor *d_inject_logits = ds4_gpu_tensor_alloc(inject_count * sizeof(float));
    ds4_gpu_tensor *d_injection = ds4_gpu_tensor_alloc(inject_count * sizeof(float));
    REQUIRE(d_down && d_down_out && d_normed && d_mix_logits && d_hyper &&
            d_residual && d_mixed && d_block && d_inject_logits && d_injection,
            "hyper-connection GPU allocation");

    write_f32(d_down, down, low_count, "HC down upload");
    REQUIRE(ds4_gpu_qwen4exp_hc_down_silu_tensor(
                d_down_out, d_down, low_count, QWEN_HC),
            "HC down SiLU launch");
    read_f32(d_down_out, down_got, low_count, "HC down download");
    compare_f32("Qwen HC SiLU(down / hc_count)", down_got, down_want,
                low_count, 2.0e-6f, 2.0e-6f);

    /* The transform is intentionally in-place capable. */
    REQUIRE(ds4_gpu_qwen4exp_hc_down_silu_tensor(
                d_down, d_down, low_count, QWEN_HC),
            "HC down SiLU in-place launch");
    read_f32(d_down, down_got, low_count, "HC down in-place download");
    compare_f32("Qwen HC down SiLU in-place", down_got, down_want,
                low_count, 2.0e-6f, 2.0e-6f);

    write_f32(d_normed, normed, hc_count, "HC normed upload");
    write_f32(d_mix_logits, mix_logits, hc_count, "HC mix logits upload");
    write_f32(d_hyper, hyper, hc_count, "HC hyper input upload");
    write_f32(d_block, block, mixed_count, "HC block output upload");
    write_f32(d_inject_logits, inject_logits, inject_count,
              "HC injection logits upload");
    REQUIRE(ds4_gpu_qwen4exp_hc_mix_inject_tensor(
                d_mixed, d_injection, d_normed, d_mix_logits,
                d_inject_logits, QWEN_ROWS, QWEN_HIDDEN, QWEN_HC),
            "HC mix/injection launch");
    read_f32(d_mixed, mixed_got, mixed_count, "HC mixed download");
    read_f32(d_injection, inject_got, inject_count, "HC injection download");
    compare_f32("Qwen HC sigmoid-weighted lane mean", mixed_got, mixed_want,
                mixed_count, 3.0e-6f, 3.0e-6f);
    compare_f32("Qwen HC 2*sigmoid(inject/hc_count)", inject_got,
                inject_want, inject_count, 2.0e-6f, 2.0e-6f);

    REQUIRE(ds4_gpu_qwen4exp_hc_residual_tensor(
                d_residual, d_hyper, d_block, d_injection,
                QWEN_ROWS, QWEN_HIDDEN, QWEN_HC),
            "HC residual injection launch");
    read_f32(d_residual, residual_got, hc_count,
             "HC residual injection download");
    compare_f32("Qwen HC residual + lane injection", residual_got,
                residual_want, hc_count, 3.0e-6f, 3.0e-6f);

    /* Final hyper_connection_mixer has no block injection projection. */
    REQUIRE(ds4_gpu_qwen4exp_hc_mix_inject_tensor(
                d_mixed, NULL, d_normed, d_mix_logits, NULL,
                QWEN_ROWS, QWEN_HIDDEN, QWEN_HC),
            "final HC mixer without injection");
    read_f32(d_mixed, mixed_got, mixed_count, "final HC mixer download");
    compare_f32("Qwen final HC no-injection path", mixed_got, mixed_want,
                mixed_count, 3.0e-6f, 3.0e-6f);
    REQUIRE(!ds4_gpu_qwen4exp_hc_mix_inject_tensor(
                d_mixed, d_injection, d_normed, d_mix_logits, NULL,
                QWEN_ROWS, QWEN_HIDDEN, QWEN_HC),
            "HC rejects half-present injection pair");

    ds4_gpu_tensor_free(d_injection);
    ds4_gpu_tensor_free(d_inject_logits);
    ds4_gpu_tensor_free(d_block);
    ds4_gpu_tensor_free(d_mixed);
    ds4_gpu_tensor_free(d_residual);
    ds4_gpu_tensor_free(d_hyper);
    ds4_gpu_tensor_free(d_mix_logits);
    ds4_gpu_tensor_free(d_normed);
    ds4_gpu_tensor_free(d_down_out);
    ds4_gpu_tensor_free(d_down);
    free(inject_got);
    free(inject_want);
    free(inject_logits);
    free(block);
    free(mixed_got);
    free(mixed_want);
    free(residual_got);
    free(residual_want);
    free(hyper);
    free(mix_logits);
    free(normed);
    free(down_got);
    free(down_want);
    free(down);
}

int main(void) {
    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(unsetenv("DS4_CUDA_COPY_MODEL") == 0,
            "disable whole-map test copy");

    const uint64_t weight_offset = 4096u;
    const uint64_t weight_bytes = (uint64_t)QWEN_WIDTH * sizeof(float);
    const uint64_t model_size = (weight_offset + weight_bytes + 4095u) & ~4095ull;
    void *model_map = NULL;
    REQUIRE(posix_memalign(&model_map, 4096u, (size_t)model_size) == 0,
            "aligned Qwen primitive model map");
    memset(model_map, 0, (size_t)model_size);
    REQUIRE(ds4_gpu_set_model_map(model_map, model_size),
            "register Qwen primitive model map");

    puts("== Qwen3.8-Flash-Next CUDA primitives ==");
    test_group_rms_norm(model_map, model_size, weight_offset);
    test_hyper_connection();

    REQUIRE(ds4_gpu_synchronize(), "final CUDA synchronization");
    ds4_gpu_unregister_model_map(model_map);
    free(model_map);
    ds4_gpu_cleanup();
    puts("all Qwen3.8 primitive checks passed");
    return 0;
}
