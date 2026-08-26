/* H200 parity for the Qwen3.8 PLE compute path after SSD row gathering:
 * BF16 activation promotion, signed-sqrt gating, and the causal dilated
 * depthwise convolution with persistent nine-token state. */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    PLE_ROWS = 17,
    PLE_GATE_ROWS = 3,
    PLE_HIDDEN = 2560,
    PLE_HC = 4,
    PLE_WIDTH = PLE_HIDDEN * PLE_HC,
    PLE_KERNEL = 4,
    PLE_DILATION = 3,
    PLE_STATE_LEN = (PLE_KERNEL - 1) * PLE_DILATION,
};

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint16_t f32_to_bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(rounded >> 16u);
}

static float bf16_bits_to_f32(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16u;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float sigmoid_ref(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static void compare_f32(const char *name, const float *got, const float *want,
                        uint64_t count, float atol, float rtol) {
    double worst_ratio = 0.0;
    float worst_abs = 0.0f;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < count; i++) {
        const float abs_error = fabsf(got[i] - want[i]);
        const float limit = atol + rtol * fabsf(want[i]);
        const double ratio = limit > 0.0f
            ? (double)abs_error / limit
            : (abs_error == 0.0f ? 0.0 : INFINITY);
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
            worst_abs = abs_error;
            worst_i = i;
        }
    }
    if (worst_ratio > 1.0) {
        fprintf(stderr,
                "FAIL: %s at %llu got %.9g want %.9g abs %.9g (%.3fx limit)\n",
                name, (unsigned long long)worst_i, got[worst_i], want[worst_i],
                worst_abs, worst_ratio);
        exit(1);
    }
    printf("%-44s pass (worst abs %.3g, %.3fx limit)\n",
           name, worst_abs, worst_ratio);
}

static void upload_f32(ds4_gpu_tensor *dst, const float *src, uint64_t count,
                       const char *what) {
    REQUIRE(ds4_gpu_tensor_write(dst, 0, src, count * sizeof(*src)), what);
}

static void download_f32(const ds4_gpu_tensor *src, float *dst, uint64_t count,
                         const char *what) {
    REQUIRE(ds4_gpu_tensor_read(src, 0, dst, count * sizeof(*dst)), what);
}

static void test_bf16_promote(void) {
    const uint64_t count = 4097u;
    uint16_t *input = (uint16_t *)malloc(count * sizeof(*input));
    float *want = (float *)malloc(count * sizeof(*want));
    float *got = (float *)malloc(count * sizeof(*got));
    REQUIRE(input && want && got, "BF16 promote host allocation");
    for (uint64_t i = 0; i < count; i++) {
        const float value = 7.0f * sinf((float)(i + 1u) * 0.019f) +
                            (float)((int)(i % 31u) - 15) * 0.0031f;
        input[i] = f32_to_bf16_bits(value);
        want[i] = bf16_bits_to_f32(input[i]);
    }
    ds4_gpu_tensor *din = ds4_gpu_tensor_alloc(count * sizeof(uint16_t));
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(count * sizeof(float));
    REQUIRE(din && dout, "BF16 promote GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(din, 0, input, count * sizeof(uint16_t)),
            "BF16 promote upload");
    REQUIRE(ds4_gpu_qwen4exp_bf16_to_f32_tensor(dout, din, count),
            "BF16 promote launch");
    download_f32(dout, got, count, "BF16 promote download");
    for (uint64_t i = 0; i < count; i++) {
        uint32_t gb, wb;
        memcpy(&gb, &got[i], sizeof(gb));
        memcpy(&wb, &want[i], sizeof(wb));
        REQUIRE(gb == wb, "BF16 promotion is bit exact");
    }
    printf("%-44s pass (%llu exact values)\n",
           "SSD PLE BF16 gather promotion", (unsigned long long)count);
    REQUIRE(!ds4_gpu_qwen4exp_bf16_to_f32_tensor(dout, din, 0),
            "BF16 promote rejects zero count");
    ds4_gpu_tensor_free(dout);
    ds4_gpu_tensor_free(din);
    free(got);
    free(want);
    free(input);
}

static void test_ple_gate(void) {
    const uint64_t hc_values =
        (uint64_t)PLE_GATE_ROWS * PLE_HC * PLE_HIDDEN;
    const uint64_t value_count =
        (uint64_t)PLE_GATE_ROWS * PLE_HIDDEN;
    const uint64_t lane_count = (uint64_t)PLE_GATE_ROWS * PLE_HC;
    float *key = (float *)malloc(hc_values * sizeof(*key));
    float *query = (float *)malloc(hc_values * sizeof(*query));
    float *value = (float *)malloc(value_count * sizeof(*value));
    float *gated_want = (float *)malloc(hc_values * sizeof(*gated_want));
    float *gated_got = (float *)malloc(hc_values * sizeof(*gated_got));
    float *gate_want = (float *)malloc(lane_count * sizeof(*gate_want));
    float *gate_got = (float *)malloc(lane_count * sizeof(*gate_got));
    REQUIRE(key && query && value && gated_want && gated_got && gate_want &&
            gate_got, "PLE gate host allocation");

    for (uint64_t i = 0; i < hc_values; i++) {
        key[i] = 0.8f * sinf((float)(i + 3u) * 0.0061f) +
                 0.1f * cosf((float)(i + 17u) * 0.013f);
        query[i] = 0.7f * cosf((float)(i + 5u) * 0.0043f) -
                   0.13f * sinf((float)(i + 23u) * 0.011f);
    }
    for (uint64_t i = 0; i < value_count; i++)
        value[i] = 0.6f * sinf((float)(i + 7u) * 0.017f);

    for (uint32_t row = 0; row < PLE_GATE_ROWS; row++) {
        for (uint32_t hc = 0; hc < PLE_HC; hc++) {
            const uint64_t lane = (uint64_t)row * PLE_HC + hc;
            const uint64_t base = lane * PLE_HIDDEN;
            double dot = 0.0;
            for (uint32_t d = 0; d < PLE_HIDDEN; d++)
                dot += (double)key[base + d] * query[base + d];
            const float raw = (float)(dot / sqrt((double)PLE_HIDDEN));
            const float sign = (float)((raw > 0.0f) - (raw < 0.0f));
            const float transformed = sqrtf(fmaxf(fabsf(raw), 1.0e-6f)) * sign;
            gate_want[lane] = transformed;
            const float gate = sigmoid_ref(transformed);
            for (uint32_t d = 0; d < PLE_HIDDEN; d++)
                gated_want[base + d] =
                    gate * value[(uint64_t)row * PLE_HIDDEN + d];
        }
    }

    ds4_gpu_tensor *dkey = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dquery = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dvalue = ds4_gpu_tensor_alloc(value_count * sizeof(float));
    ds4_gpu_tensor *dgated = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dgate = ds4_gpu_tensor_alloc(lane_count * sizeof(float));
    REQUIRE(dkey && dquery && dvalue && dgated && dgate,
            "PLE gate GPU allocation");
    upload_f32(dkey, key, hc_values, "PLE key upload");
    upload_f32(dquery, query, hc_values, "PLE query upload");
    upload_f32(dvalue, value, value_count, "PLE value upload");
    REQUIRE(ds4_gpu_qwen4exp_ple_gate_tensor(
                dgated, dgate, dkey, dquery, dvalue,
                PLE_GATE_ROWS, PLE_HIDDEN, PLE_HC),
            "PLE gate launch");
    download_f32(dgated, gated_got, hc_values, "PLE gated-value download");
    download_f32(dgate, gate_got, lane_count, "PLE transformed-gate download");
    compare_f32("PLE signed-sqrt transformed gate", gate_got, gate_want,
                lane_count, 2.0e-5f, 2.0e-5f);
    compare_f32("PLE sigmoid gate times shared value", gated_got, gated_want,
                hc_values, 4.0e-6f, 4.0e-6f);

    /* Telemetry gate output is optional, but in-place value overwrite is not
     * safe because all four lanes broadcast the same value row. */
    REQUIRE(ds4_gpu_qwen4exp_ple_gate_tensor(
                dgated, NULL, dkey, dquery, dvalue,
                PLE_GATE_ROWS, PLE_HIDDEN, PLE_HC),
            "PLE gate without telemetry output");
    REQUIRE(!ds4_gpu_qwen4exp_ple_gate_tensor(
                dvalue, NULL, dkey, dquery, dvalue,
                PLE_GATE_ROWS, PLE_HIDDEN, PLE_HC),
            "PLE gate rejects value/output alias");

    ds4_gpu_tensor_free(dgate);
    ds4_gpu_tensor_free(dgated);
    ds4_gpu_tensor_free(dvalue);
    ds4_gpu_tensor_free(dquery);
    ds4_gpu_tensor_free(dkey);
    free(gate_got);
    free(gate_want);
    free(gated_got);
    free(gated_want);
    free(value);
    free(query);
    free(key);
}

static void conv_reference(float *out, float *state, const float *input,
                           const float *weight, uint32_t rows) {
    for (uint32_t token = 0; token < rows; token++) {
        for (uint32_t channel = 0; channel < PLE_WIDTH; channel++) {
            float sum = 0.0f;
            for (uint32_t tap = 0; tap < PLE_KERNEL; tap++) {
                const int32_t source_token = (int32_t)token - PLE_STATE_LEN +
                                             (int32_t)tap * PLE_DILATION;
                const float source = source_token >= 0
                    ? input[(uint64_t)source_token * PLE_WIDTH + channel]
                    : state[(uint64_t)channel * PLE_STATE_LEN +
                            (uint32_t)(PLE_STATE_LEN + source_token)];
                sum += source * weight[(uint64_t)channel * PLE_KERNEL + tap];
            }
            out[(uint64_t)token * PLE_WIDTH + channel] =
                sum * sigmoid_ref(sum);
        }
    }
    for (uint32_t channel = 0; channel < PLE_WIDTH; channel++) {
        float prior[PLE_STATE_LEN];
        for (uint32_t slot = 0; slot < PLE_STATE_LEN; slot++)
            prior[slot] = state[(uint64_t)channel * PLE_STATE_LEN + slot];
        for (uint32_t slot = 0; slot < PLE_STATE_LEN; slot++) {
            const int32_t combined = (int32_t)rows + slot - PLE_STATE_LEN;
            state[(uint64_t)channel * PLE_STATE_LEN + slot] = combined >= 0
                ? input[(uint64_t)combined * PLE_WIDTH + channel]
                : prior[PLE_STATE_LEN + combined];
        }
    }
}

static void run_conv_chunks(ds4_gpu_tensor *output, ds4_gpu_tensor *state,
                            ds4_gpu_tensor *input, const uint32_t *chunks,
                            size_t chunk_count, const void *model_map,
                            uint64_t model_size, uint64_t weight_offset) {
    uint32_t row = 0;
    for (size_t i = 0; i < chunk_count; i++) {
        const uint32_t rows = chunks[i];
        const uint64_t bytes = (uint64_t)rows * PLE_WIDTH * sizeof(float);
        const uint64_t offset = (uint64_t)row * PLE_WIDTH * sizeof(float);
        ds4_gpu_tensor *input_view = ds4_gpu_tensor_view(input, offset, bytes);
        ds4_gpu_tensor *output_view = ds4_gpu_tensor_view(output, offset, bytes);
        REQUIRE(input_view && output_view, "PLE convolution chunk views");
        REQUIRE(ds4_gpu_qwen4exp_ple_conv_tensor(
                    output_view, state, input_view, model_map, model_size,
                    weight_offset, rows, PLE_WIDTH, PLE_KERNEL, PLE_DILATION),
                "PLE convolution chunk launch");
        ds4_gpu_tensor_free(output_view);
        ds4_gpu_tensor_free(input_view);
        row += rows;
    }
    REQUIRE(row == PLE_ROWS, "PLE convolution chunks cover all rows");
}

static void test_ple_convolution(void *model_map, uint64_t model_size,
                                 uint64_t weight_offset) {
    const uint64_t count = (uint64_t)PLE_ROWS * PLE_WIDTH;
    const uint64_t weight_count = (uint64_t)PLE_WIDTH * PLE_KERNEL;
    const uint64_t state_count = (uint64_t)PLE_WIDTH * PLE_STATE_LEN;
    uint16_t *weight_bits =
        (uint16_t *)((unsigned char *)model_map + weight_offset);
    float *weight = (float *)malloc(weight_count * sizeof(*weight));
    float *input = (float *)malloc(count * sizeof(*input));
    float *want = (float *)malloc(count * sizeof(*want));
    float *got = (float *)malloc(count * sizeof(*got));
    float *state_want = (float *)calloc(state_count, sizeof(*state_want));
    float *state_got = (float *)malloc(state_count * sizeof(*state_got));
    REQUIRE(weight && input && want && got && state_want && state_got,
            "PLE convolution host allocation");

    for (uint64_t i = 0; i < weight_count; i++) {
        const float source = 0.22f * sinf((float)(i + 1u) * 0.031f) +
                             (float)((int)(i % 7u) - 3) * 0.009f;
        weight_bits[i] = f32_to_bf16_bits(source);
        weight[i] = bf16_bits_to_f32(weight_bits[i]);
    }
    for (uint64_t i = 0; i < count; i++)
        input[i] = 0.75f * sinf((float)(i + 5u) * 0.0041f) +
                   0.18f * cosf((float)(i + 19u) * 0.013f);
    conv_reference(want, state_want, input, weight, PLE_ROWS);

    ds4_gpu_tensor *dinput = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *doutput = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *dstate = ds4_gpu_tensor_alloc(state_count * sizeof(float));
    REQUIRE(dinput && doutput && dstate, "PLE convolution GPU allocation");
    upload_f32(dinput, input, count, "PLE convolution input upload");

    REQUIRE(ds4_gpu_tensor_fill_f32(dstate, 0.0f, state_count),
            "PLE convolution full state reset");
    REQUIRE(ds4_gpu_qwen4exp_ple_conv_tensor(
                doutput, dstate, dinput, model_map, model_size, weight_offset,
                PLE_ROWS, PLE_WIDTH, PLE_KERNEL, PLE_DILATION),
            "PLE convolution full-prefill launch");
    download_f32(doutput, got, count, "PLE convolution full-prefill download");
    compare_f32("PLE dilated convolution full prefill", got, want, count,
                4.0e-6f, 4.0e-6f);
    download_f32(dstate, state_got, state_count,
                 "PLE convolution full state download");
    compare_f32("PLE convolution full final state", state_got, state_want,
                state_count, 0.0f, 0.0f);

    const uint32_t irregular_chunks[] = {2u, 5u, 1u, 9u};
    REQUIRE(ds4_gpu_tensor_fill_f32(dstate, 0.0f, state_count),
            "PLE convolution chunked state reset");
    run_conv_chunks(doutput, dstate, dinput, irregular_chunks,
                    sizeof(irregular_chunks) / sizeof(irregular_chunks[0]),
                    model_map, model_size, weight_offset);
    download_f32(doutput, got, count, "PLE convolution chunked download");
    compare_f32("PLE convolution irregular chunk parity", got, want, count,
                4.0e-6f, 4.0e-6f);
    download_f32(dstate, state_got, state_count,
                 "PLE convolution chunked state download");
    compare_f32("PLE convolution chunked final state", state_got, state_want,
                state_count, 0.0f, 0.0f);

    uint32_t decode_chunks[PLE_ROWS];
    for (uint32_t i = 0; i < PLE_ROWS; i++) decode_chunks[i] = 1u;
    REQUIRE(ds4_gpu_tensor_fill_f32(dstate, 0.0f, state_count),
            "PLE convolution decode state reset");
    run_conv_chunks(doutput, dstate, dinput, decode_chunks, PLE_ROWS,
                    model_map, model_size, weight_offset);
    download_f32(doutput, got, count, "PLE convolution decode download");
    compare_f32("PLE convolution token-decode parity", got, want, count,
                4.0e-6f, 4.0e-6f);
    download_f32(dstate, state_got, state_count,
                 "PLE convolution decode state download");
    compare_f32("PLE convolution decode final state", state_got, state_want,
                state_count, 0.0f, 0.0f);

    REQUIRE(!ds4_gpu_qwen4exp_ple_conv_tensor(
                dinput, dstate, dinput, model_map, model_size, weight_offset,
                PLE_ROWS, PLE_WIDTH, PLE_KERNEL, PLE_DILATION),
            "PLE convolution rejects input/output alias");
    REQUIRE(!ds4_gpu_qwen4exp_ple_conv_tensor(
                doutput, dstate, dinput, model_map, model_size, weight_offset,
                PLE_ROWS, PLE_WIDTH, PLE_KERNEL, 0),
            "PLE convolution rejects zero dilation");

    ds4_gpu_tensor_free(dstate);
    ds4_gpu_tensor_free(doutput);
    ds4_gpu_tensor_free(dinput);
    free(state_got);
    free(state_want);
    free(got);
    free(want);
    free(input);
    free(weight);
}

int main(void) {
    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(unsetenv("DS4_CUDA_COPY_MODEL") == 0,
            "disable whole-map test copy");

    const uint64_t weight_offset = 4096u;
    const uint64_t weight_bytes =
        (uint64_t)PLE_WIDTH * PLE_KERNEL * sizeof(uint16_t);
    const uint64_t model_size =
        (weight_offset + weight_bytes + 4095u) & ~4095ull;
    void *model_map = NULL;
    REQUIRE(posix_memalign(&model_map, 4096u, (size_t)model_size) == 0,
            "aligned PLE compute model map");
    memset(model_map, 0, (size_t)model_size);
    REQUIRE(ds4_gpu_set_model_map(model_map, model_size),
            "register PLE compute model map");

    puts("== Qwen3.8-Flash-Next PLE compute primitives ==");
    test_bf16_promote();
    test_ple_gate();
    test_ple_convolution(model_map, model_size, weight_offset);

    REQUIRE(ds4_gpu_synchronize(), "final CUDA synchronization");
    ds4_gpu_unregister_model_map(model_map);
    free(model_map);
    ds4_gpu_cleanup();
    puts("all Qwen3.8 PLE compute checks passed");
    return 0;
}
