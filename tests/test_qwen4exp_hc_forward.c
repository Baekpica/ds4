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
    ds4_tensor dense = make_weight(&cursor, WIDTH, LOW);
    const uint64_t model_size = align4k(cursor);
    unsigned char *map = NULL;
    REQUIRE(posix_memalign((void **)&map, 4096u, (size_t)model_size) == 0,
            "model fixture allocation");
    memset(map, 0, (size_t)model_size);

    uint16_t *down_data = (uint16_t *)(map + down.abs_offset);
    uint16_t *up_data = (uint16_t *)(map + up.abs_offset);
    uint16_t *inject_data = (uint16_t *)(map + inject.abs_offset);
    uint16_t *dense_data = (uint16_t *)(map + dense.abs_offset);
    for (uint32_t low = 0; low < LOW; low++)
        down_data[(uint64_t)low * WIDTH + low] = bf16(1.0f);
    for (uint32_t out = 0; out < WIDTH; out++)
        up_data[(uint64_t)out * LOW + out % LOW] = bf16(1.0f);
    for (uint32_t lane = 0; lane < HC; lane++)
        inject_data[(uint64_t)lane * WIDTH + (uint64_t)lane * HIDDEN] = bf16(1.0f);
    for (uint64_t i = 0u; i < dense.elements; i++)
        dense_data[i] = bf16((float)((int)(i % 29u) - 14) * 0.001f);

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
    float *scalar_got = malloc(hc_values * sizeof(*scalar_got));
    float *scalar_mixed = malloc(block_values * sizeof(*scalar_mixed));
    REQUIRE(input && block && got && want && mixed_got && mixed_want &&
            scalar_got && scalar_mixed,
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
    ds4_gpu_tensor *dense_batch = ds4_gpu_tensor_alloc(
        (uint64_t)ROWS * LOW * sizeof(float));
    ds4_gpu_tensor *dense_scalar = ds4_gpu_tensor_alloc(
        (uint64_t)ROWS * LOW * sizeof(float));
    ds4_qwen_hc_ws ws;
    ds4_qwen_hc_ws scalar_ws;
    REQUIRE(dinput && dblock && dout && dense_batch && dense_scalar &&
            qwen4exp_hc_ws_alloc(&ws, ROWS, HIDDEN, HC, LOW) &&
            qwen4exp_hc_ws_alloc(&scalar_ws, 1u, HIDDEN, HC, LOW),
            "HC workspace");
    REQUIRE(ds4_gpu_tensor_write(dinput, 0, input, hc_values * sizeof(float)) &&
            ds4_gpu_tensor_write(dblock, 0, block, block_values * sizeof(float)),
            "input upload");
    REQUIRE(qwen4exp_hc_begin(
                &ws, &model, &weights, dinput, ROWS, true), "HC begin");
    REQUIRE(ds4_gpu_tensor_read(ws.mixed, 0, mixed_got,
                                block_values * sizeof(float)),
            "mixed-input readback");
    REQUIRE(qwen4exp_hc_finish(&ws, dinput, dblock, dout, ROWS), "HC finish");
    REQUIRE(ds4_gpu_tensor_read(dout, 0, got, hc_values * sizeof(float)), "output readback");
    for (uint32_t row = 0u; row < ROWS; row++) {
        ds4_gpu_tensor *input_row = ds4_gpu_tensor_view(
            dinput, (uint64_t)row * WIDTH * sizeof(float),
            (uint64_t)WIDTH * sizeof(float));
        ds4_gpu_tensor *block_row = ds4_gpu_tensor_view(
            dblock, (uint64_t)row * HIDDEN * sizeof(float),
            (uint64_t)HIDDEN * sizeof(float));
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            dout, (uint64_t)row * WIDTH * sizeof(float),
            (uint64_t)WIDTH * sizeof(float));
        REQUIRE(input_row && block_row && out_row &&
                qwen4exp_hc_begin(
                    &scalar_ws, &model, &weights, input_row, 1u, true) &&
                ds4_gpu_tensor_read(
                    scalar_ws.mixed, 0,
                    scalar_mixed + (uint64_t)row * HIDDEN,
                    (uint64_t)HIDDEN * sizeof(float)) &&
                qwen4exp_hc_finish(
                    &scalar_ws, input_row, block_row, out_row, 1u),
                "scalar-row HC");
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(block_row);
        ds4_gpu_tensor_free(input_row);
    }
    REQUIRE(ds4_gpu_tensor_read(
                dout, 0, scalar_got, hc_values * sizeof(float)),
            "scalar-row output readback");
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
    REQUIRE(memcmp(mixed_got, scalar_mixed,
                   block_values * sizeof(float)) == 0,
            "two-row HC mixed output differs from scalar rows");
    REQUIRE(memcmp(got, scalar_got, hc_values * sizeof(float)) == 0,
            "two-row HC residual differs from scalar rows");

    REQUIRE(ds4_gpu_matmul_bf16_stable_rows_tensor(
                dense_batch, map, model_size, dense.abs_offset,
                WIDTH, LOW, dinput, ROWS),
            "dense two-row BF16 projection");
    for (uint32_t row = 0u; row < ROWS; row++) {
        ds4_gpu_tensor *input_row = ds4_gpu_tensor_view(
            dinput, (uint64_t)row * WIDTH * sizeof(float),
            (uint64_t)WIDTH * sizeof(float));
        ds4_gpu_tensor *output_row = ds4_gpu_tensor_view(
            dense_scalar, (uint64_t)row * LOW * sizeof(float),
            (uint64_t)LOW * sizeof(float));
        REQUIRE(input_row && output_row &&
                ds4_gpu_matmul_bf16_stable_rows_tensor(
                    output_row, map, model_size, dense.abs_offset,
                    WIDTH, LOW, input_row, 1u),
                "dense scalar-row BF16 projection");
        ds4_gpu_tensor_free(output_row);
        ds4_gpu_tensor_free(input_row);
    }
    float dense_batch_host[ROWS * LOW];
    float dense_scalar_host[ROWS * LOW];
    REQUIRE(ds4_gpu_tensor_read(
                dense_batch, 0u, dense_batch_host,
                sizeof(dense_batch_host)) &&
            ds4_gpu_tensor_read(
                dense_scalar, 0u, dense_scalar_host,
                sizeof(dense_scalar_host)),
            "dense BF16 projection readback");
    REQUIRE(memcmp(dense_batch_host, dense_scalar_host,
                   sizeof(dense_batch_host)) == 0,
            "dense two-row BF16 projection differs from scalar rows");

    qwen4exp_hc_ws_free(&scalar_ws);
    qwen4exp_hc_ws_free(&ws);
    ds4_gpu_tensor_free(dense_scalar);
    ds4_gpu_tensor_free(dense_batch);
    ds4_gpu_tensor_free(dout);
    ds4_gpu_tensor_free(dblock);
    ds4_gpu_tensor_free(dinput);
    ds4_gpu_unregister_model_map(map);
    ds4_gpu_cleanup();
    free(scalar_mixed); free(scalar_got);
    free(mixed_want); free(mixed_got); free(want); free(got);
    free(block); free(input); free(map);
    return 0;
}
