/* Projection-to-residual integration check for Qwen4Exp hyper-connections. */
#include "../ds4.c"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ROWS = 2,
    HIDDEN = 2560,
    HC = 4,
    WIDTH = HIDDEN * HC,
    LOW = 320,
};

#define REQUIRE(c, m) do {                                                     \
    if (!(c)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", m, __FILE__, __LINE__); exit(1); } \
} while (0)

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static uint64_t align4k(uint64_t value) {
    return (value + 4095u) & ~UINT64_C(4095);
}

static ds4_tensor make_weight(uint64_t *cursor, uint64_t d0, uint64_t d1) {
    ds4_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.ndim = d1 ? 2u : 1u;
    tensor.dim[0] = d0;
    tensor.dim[1] = d1;
    tensor.type = d1 ? DS4_TENSOR_BF16 : DS4_TENSOR_F32;
    tensor.elements = d0 * (d1 ? d1 : 1u);
    tensor.bytes = tensor.elements * (d1 ? sizeof(uint16_t) : sizeof(float));
    tensor.abs_offset = tensor.rel_offset = align4k(*cursor);
    *cursor = tensor.abs_offset + tensor.bytes;
    return tensor;
}

static float sigmoidf_ref(float value) {
    return 1.0f / (1.0f + expf(-value));
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_QWEN38_FLASH_NEXT;
    uint64_t cursor = 4096u;
    ds4_tensor norm = make_weight(&cursor, WIDTH, 0u);
    ds4_tensor down = make_weight(&cursor, WIDTH, LOW);
    ds4_tensor up = make_weight(&cursor, LOW, WIDTH);
    ds4_tensor inject = make_weight(&cursor, WIDTH, HC);
    const uint64_t model_size = align4k(cursor);
    unsigned char *map = NULL;
    REQUIRE(posix_memalign((void **)&map, 4096u, (size_t)model_size) == 0,
            "model fixture allocation");
    memset(map, 0, (size_t)model_size);

    uint16_t *down_data = (uint16_t *)(map + down.abs_offset);
    uint16_t *up_data = (uint16_t *)(map + up.abs_offset);
    uint16_t *inject_data = (uint16_t *)(map + inject.abs_offset);
    for (uint32_t low = 0; low < LOW; low++)
        down_data[(uint64_t)low * WIDTH + low] = bf16(1.0f);
    for (uint32_t out = 0; out < WIDTH; out++)
        up_data[(uint64_t)out * LOW + out % LOW] = bf16(1.0f);
    for (uint32_t lane = 0; lane < HC; lane++)
        inject_data[(uint64_t)lane * WIDTH + (uint64_t)lane * HIDDEN] = bf16(1.0f);

    ds4_model model;
    memset(&model, 0, sizeof(model));
    model.fd = -1;
    model.map = map;
    model.size = model_size;
    ds4_qwen_hc_weights weights = {
        .norm = &norm,
        .mix_down = &down,
        .mix_up = &up,
        .inject = &inject,
    };

    const uint64_t hc_values = (uint64_t)ROWS * WIDTH;
    const uint64_t block_values = (uint64_t)ROWS * HIDDEN;
    float *input = malloc(hc_values * sizeof(*input));
    float *block = malloc(block_values * sizeof(*block));
    float *got = malloc(hc_values * sizeof(*got));
    float *want = malloc(hc_values * sizeof(*want));
    float *mixed_got = malloc(block_values * sizeof(*mixed_got));
    float *mixed_want = malloc(block_values * sizeof(*mixed_want));
    REQUIRE(input && block && got && want && mixed_got && mixed_want,
            "host buffers");
    for (uint64_t i = 0; i < hc_values; i++)
        input[i] = 0.8f * sinf((float)(i + 1u) * 0.003f) + 0.2f;
    for (uint64_t i = 0; i < block_values; i++)
        block[i] = 0.4f * cosf((float)(i + 3u) * 0.007f);

    for (uint32_t row = 0; row < ROWS; row++) {
        float normed[WIDTH];
        for (uint32_t lane = 0; lane < HC; lane++) {
            const uint64_t base = (uint64_t)row * WIDTH + (uint64_t)lane * HIDDEN;
            double sum = 0.0;
            for (uint32_t d = 0; d < HIDDEN; d++) sum += (double)input[base + d] * input[base + d];
            const float scale = 1.0f / sqrtf((float)(sum / HIDDEN) + DS4_RMS_EPS);
            for (uint32_t d = 0; d < HIDDEN; d++) normed[(uint64_t)lane * HIDDEN + d] = input[base + d] * scale;
        }
        float low[LOW];
        for (uint32_t i = 0; i < LOW; i++) {
            const float projected = normed[i] / HC;
            low[i] = projected * sigmoidf_ref(projected);
        }
        float injection[HC];
        for (uint32_t lane = 0; lane < HC; lane++)
            injection[lane] = 2.0f * sigmoidf_ref(normed[(uint64_t)lane * HIDDEN] / HC);
        for (uint32_t lane = 0; lane < HC; lane++) {
            for (uint32_t d = 0; d < HIDDEN; d++) {
                float mixed = 0.0f;
                for (uint32_t source = 0; source < HC; source++) {
                    const uint32_t at = source * HIDDEN + d;
                    mixed += sigmoidf_ref(low[at % LOW]) * normed[at];
                }
                mixed /= HC;
                mixed_want[(uint64_t)row * HIDDEN + d] = mixed;
                const uint64_t at = (uint64_t)row * WIDTH + (uint64_t)lane * HIDDEN + d;
                want[at] = input[at] + block[(uint64_t)row * HIDDEN + d] * injection[lane];
            }
        }
    }

    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(ds4_gpu_set_model_map(map, model_size), "model map registration");
    ds4_gpu_tensor *dinput = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dblock = ds4_gpu_tensor_alloc(block_values * sizeof(float));
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_qwen_hc_ws ws;
    REQUIRE(dinput && dblock && dout &&
            qwen4exp_hc_ws_alloc(&ws, ROWS, HIDDEN, HC, LOW), "HC workspace");
    REQUIRE(ds4_gpu_tensor_write(dinput, 0, input, hc_values * sizeof(float)) &&
            ds4_gpu_tensor_write(dblock, 0, block, block_values * sizeof(float)),
            "input upload");
    REQUIRE(qwen4exp_hc_begin(&ws, &model, &weights, dinput, ROWS), "HC begin");
    REQUIRE(ds4_gpu_tensor_read(ws.mixed, 0, mixed_got,
                                block_values * sizeof(float)),
            "mixed-input readback");
    REQUIRE(qwen4exp_hc_finish(&ws, dinput, dblock, dout, ROWS), "HC finish");
    REQUIRE(ds4_gpu_tensor_read(dout, 0, got, hc_values * sizeof(float)), "output readback");
    float mixed_worst = 0.0f, worst = 0.0f;
    for (uint64_t i = 0; i < block_values; i++) {
        const float error = fabsf(mixed_got[i] - mixed_want[i]);
        if (error > mixed_worst) mixed_worst = error;
    }
    for (uint64_t i = 0; i < hc_values; i++) {
        const float error = fabsf(got[i] - want[i]);
        if (error > worst) worst = error;
    }
    printf("Qwen HC projection/residual integration passed (max %.3g / %.3g)\n",
           mixed_worst, worst);
    REQUIRE(mixed_worst < 2.0e-4f, "integrated HC projection parity");
    REQUIRE(worst < 2.0e-4f, "integrated HC residual parity");

    qwen4exp_hc_ws_free(&ws);
    ds4_gpu_tensor_free(dout);
    ds4_gpu_tensor_free(dblock);
    ds4_gpu_tensor_free(dinput);
    ds4_gpu_unregister_model_map(map);
    ds4_gpu_cleanup();
    free(mixed_want); free(mixed_got); free(want); free(got);
    free(block); free(input); free(map);
    return 0;
}
