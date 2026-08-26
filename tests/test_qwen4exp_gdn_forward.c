/* Real-artifact H200 parity for a complete Qwen3.8 Gated DeltaNet layer.
 * The nine layer-0 tensors are copied into a compact ~60 MiB fixture so the
 * published 90.96 GiB backbone remains mmap-only. */
#include "../ds4.c"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

enum {
    TEST_LAYER = 0,
    TEST_ROWS = 3,
    TEST_HIDDEN = 2560,
    TEST_KEY_HEADS = 16,
    TEST_VALUE_HEADS = 48,
    TEST_HEAD_DIM = 128,
    TEST_KEY_DIM = TEST_KEY_HEADS * TEST_HEAD_DIM,
    TEST_VALUE_DIM = TEST_VALUE_HEADS * TEST_HEAD_DIM,
    TEST_CONV_DIM = 2 * TEST_KEY_DIM + TEST_VALUE_DIM,
    TEST_CONV_KERNEL = 4,
};

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint64_t gdn_align(uint64_t value) {
    return (value + 4095u) & ~UINT64_C(4095);
}

static ds4_tensor gdn_copy_tensor(
        unsigned char *fixture, uint64_t fixture_bytes, uint64_t *cursor,
        const ds4_model *model, const ds4_tensor *source) {
    const uint64_t offset = gdn_align(*cursor);
    REQUIRE(offset <= fixture_bytes && source->bytes <= fixture_bytes - offset,
            "GDN fixture tensor range");
    memcpy(fixture + offset, tensor_data(model, source), (size_t)source->bytes);
    ds4_tensor copy = *source;
    copy.rel_offset = offset;
    copy.abs_offset = offset;
    *cursor = offset + source->bytes;
    return copy;
}

static float bf16_value(const uint16_t bits) {
    const uint32_t word = (uint32_t)bits << 16u;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

static float gdn_sigmoid(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x))
                     : expf(x) / (1.0f + expf(x));
}

static float gdn_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static void gdn_compare(const char *name, const float *got,
                        const float *want, uint64_t count,
                        double max_rel_rms, double max_one_minus_cos,
                        float max_abs_limit) {
    double error2 = 0.0;
    double ref2 = 0.0;
    double got2 = 0.0;
    double dot = 0.0;
    float max_abs = 0.0f;
    uint64_t worst = 0u;
    for (uint64_t i = 0; i < count; i++) {
        const double error = (double)got[i] - want[i];
        error2 += error * error;
        ref2 += (double)want[i] * want[i];
        got2 += (double)got[i] * got[i];
        dot += (double)got[i] * want[i];
        const float absolute = fabsf(got[i] - want[i]);
        if (absolute > max_abs) {
            max_abs = absolute;
            worst = i;
        }
    }
    const double rel_rms = sqrt(error2 / fmax(ref2, 1.0e-30));
    const double one_minus_cos = ref2 > 0.0 && got2 > 0.0
        ? 1.0 - dot / sqrt(ref2 * got2) : 0.0;
    printf("%-49s rel_rms %.4e 1-cos %.4e max %.5g @%llu\n",
           name, rel_rms, one_minus_cos, max_abs,
           (unsigned long long)worst);
    REQUIRE(rel_rms <= max_rel_rms, "real GDN relative RMS gate");
    REQUIRE(one_minus_cos <= max_one_minus_cos,
            "real GDN cosine gate");
    REQUIRE(max_abs <= max_abs_limit, "real GDN maximum error gate");
}

static void gdn_matvec_rows(float *out, const ds4_model *model,
                            const ds4_tensor *weight, const float *input,
                            uint32_t rows, uint32_t in_dim,
                            uint32_t out_dim) {
    REQUIRE(weight->type == DS4_TENSOR_Q8_0, "real GDN Q8 projection type");
    for (uint32_t row = 0; row < rows; row++)
        matvec_q8_0(out + (uint64_t)row * out_dim, model, weight,
                    input + (uint64_t)row * in_dim);
}

static void gdn_conv_reference(float *out, float *state, const float *input,
                               const uint16_t *weight, uint32_t rows) {
    for (uint32_t token = 0; token < rows; token++) {
        for (uint32_t channel = 0; channel < TEST_CONV_DIM; channel++) {
            const uint64_t base = (uint64_t)channel * TEST_CONV_KERNEL;
            float sum = 0.0f;
            for (uint32_t tap = 0; tap < TEST_CONV_KERNEL - 1u; tap++)
                sum += state[base + tap + 1u] *
                       bf16_value(weight[base + tap]);
            const float current =
                input[(uint64_t)token * TEST_CONV_DIM + channel];
            sum += current * bf16_value(weight[base + 3u]);
            out[(uint64_t)token * TEST_CONV_DIM + channel] =
                sum * gdn_sigmoid(sum);
            state[base + 0u] = state[base + 1u];
            state[base + 1u] = state[base + 2u];
            state[base + 2u] = state[base + 3u];
            state[base + 3u] = current;
        }
    }
}

static void gdn_controls_reference(float *beta, float *g,
                                   const float *b, const float *a,
                                   const float *a_log,
                                   const float *dt_bias, uint32_t rows) {
    for (uint64_t i = 0; i < (uint64_t)rows * TEST_VALUE_HEADS; i++) {
        const uint32_t head = (uint32_t)(i % TEST_VALUE_HEADS);
        beta[i] = gdn_sigmoid(b[i]);
        g[i] = -expf(a_log[head]) *
               gdn_softplus(a[i] + dt_bias[head]);
    }
}

static void gdn_recurrent_reference(float *out, float *state,
                                    const float *mixed, const float *beta,
                                    const float *g, uint32_t rows) {
    const uint32_t repeat = TEST_VALUE_HEADS / TEST_KEY_HEADS;
    for (uint32_t token = 0; token < rows; token++) {
        const uint64_t rb = (uint64_t)token * TEST_CONV_DIM;
        for (uint32_t vh = 0; vh < TEST_VALUE_HEADS; vh++) {
            const uint32_t kh = vh / repeat;
            const float *q = mixed + rb + (uint64_t)kh * TEST_HEAD_DIM;
            const float *k = mixed + rb + TEST_KEY_DIM +
                             (uint64_t)kh * TEST_HEAD_DIM;
            const float *value = mixed + rb + 2u * TEST_KEY_DIM +
                                 (uint64_t)vh * TEST_HEAD_DIM;
            float q2 = 0.0f, k2 = 0.0f;
            for (uint32_t d = 0; d < TEST_HEAD_DIM; d++) {
                q2 += q[d] * q[d];
                k2 += k[d] * k[d];
            }
            const float qi = 1.0f / sqrtf(q2 + 1.0e-6f) /
                             sqrtf((float)TEST_HEAD_DIM);
            const float ki = 1.0f / sqrtf(k2 + 1.0e-6f);
            const float decay =
                expf(g[(uint64_t)token * TEST_VALUE_HEADS + vh]);
            const float bt =
                beta[(uint64_t)token * TEST_VALUE_HEADS + vh];
            const uint64_t sh =
                (uint64_t)vh * TEST_HEAD_DIM * TEST_HEAD_DIM;
            for (uint32_t v = 0; v < TEST_HEAD_DIM; v++) {
                float memory = 0.0f;
                for (uint32_t d = 0; d < TEST_HEAD_DIM; d++) {
                    const uint64_t at =
                        sh + (uint64_t)d * TEST_HEAD_DIM + v;
                    state[at] *= decay;
                    memory += state[at] * (k[d] * ki);
                }
                const float delta = (value[v] - memory) * bt;
                float result = 0.0f;
                for (uint32_t d = 0; d < TEST_HEAD_DIM; d++) {
                    const uint64_t at =
                        sh + (uint64_t)d * TEST_HEAD_DIM + v;
                    state[at] += (k[d] * ki) * delta;
                    result += state[at] * (q[d] * qi);
                }
                out[((uint64_t)token * TEST_VALUE_HEADS + vh) *
                    TEST_HEAD_DIM + v] = result;
            }
        }
    }
}

static void gdn_norm_reference(float *out, const float *core,
                               const float *z, const float *weight,
                               uint32_t rows) {
    for (uint64_t hr = 0; hr < (uint64_t)rows * TEST_VALUE_HEADS; hr++) {
        const uint64_t base = hr * TEST_HEAD_DIM;
        float sum = 0.0f;
        for (uint32_t d = 0; d < TEST_HEAD_DIM; d++)
            sum += core[base + d] * core[base + d];
        const float scale =
            1.0f / sqrtf(sum / (float)TEST_HEAD_DIM + DS4_RMS_EPS);
        for (uint32_t d = 0; d < TEST_HEAD_DIM; d++) {
            const float gate = z[base + d];
            out[base + d] = core[base + d] * scale * weight[d] *
                            gdn_sigmoid(gate);
        }
    }
}

static void gdn_run_chunked(ds4_qwen_gdn_ws *ws,
                            ds4_qwen_gdn_state *state,
                            const ds4_model *model,
                            const ds4_qwen_linear_attention_weights *weights,
                            ds4_gpu_tensor *input, ds4_gpu_tensor *output,
                            const uint32_t *chunks, uint32_t count) {
    uint32_t row = 0u;
    for (uint32_t c = 0; c < count; c++) {
        const uint64_t in_offset =
            (uint64_t)row * TEST_HIDDEN * sizeof(float);
        const uint64_t bytes =
            (uint64_t)chunks[c] * TEST_HIDDEN * sizeof(float);
        ds4_gpu_tensor *iv = ds4_gpu_tensor_view(input, in_offset, bytes);
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(output, in_offset, bytes);
        REQUIRE(iv && ov, "real GDN chunk views");
        REQUIRE(qwen4exp_gdn_forward(
                    ws, state, model, weights, iv, ov, chunks[c]),
                "real GDN chunk forward");
        ds4_gpu_tensor_free(ov);
        ds4_gpu_tensor_free(iv);
        row += chunks[c];
    }
    REQUIRE(row == TEST_ROWS, "real GDN chunks cover input");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIRST_GGUF\n", argv[0]);
        return 2;
    }
    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN4EXP,
            "Qwen4Exp model profile");
    ds4_weights all_weights;
    weights_bind(&all_weights, &model, false, 0u, DS4_N_LAYER - 1u,
                 true, false);
    const ds4_qwen_linear_attention_weights *source =
        &all_weights.layer[TEST_LAYER].qwen_linear_attn;
    REQUIRE(source->qkv && source->qkv->type == DS4_TENSOR_Q8_0 &&
            source->z->type == DS4_TENSOR_Q8_0 &&
            source->in_a->type == DS4_TENSOR_Q8_0 &&
            source->in_b->type == DS4_TENSOR_Q8_0 &&
            source->out->type == DS4_TENSOR_Q8_0 &&
            source->conv->type == DS4_TENSOR_BF16 &&
            source->a_log->type == DS4_TENSOR_F32 &&
            source->dt_bias->type == DS4_TENSOR_F32 &&
            source->norm->type == DS4_TENSOR_F32,
            "published GDN recipe types");

    const ds4_tensor *tensors[] = {
        source->qkv, source->z, source->in_a, source->in_b, source->out,
        source->conv, source->a_log, source->dt_bias, source->norm,
    };
    uint64_t fixture_bytes = 4096u;
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++)
        fixture_bytes = gdn_align(fixture_bytes) + tensors[i]->bytes;
    fixture_bytes = gdn_align(fixture_bytes);
    unsigned char *fixture = NULL;
    REQUIRE(posix_memalign((void **)&fixture, 4096u,
                           (size_t)fixture_bytes) == 0,
            "real GDN compact fixture allocation");
    memset(fixture, 0, (size_t)fixture_bytes);
    uint64_t cursor = 4096u;
    ds4_tensor qkv = gdn_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->qkv);
    ds4_tensor z_weight = gdn_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->z);
    ds4_tensor in_a = gdn_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->in_a);
    ds4_tensor in_b = gdn_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->in_b);
    ds4_tensor out_weight = gdn_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->out);
    ds4_tensor conv = gdn_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->conv);
    ds4_tensor a_log = gdn_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->a_log);
    ds4_tensor dt_bias = gdn_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->dt_bias);
    ds4_tensor norm = gdn_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->norm);
    REQUIRE(gdn_align(cursor) == fixture_bytes, "real GDN fixture census");
    ds4_qwen_linear_attention_weights weights = {
        .a_log = &a_log, .conv = &conv, .dt_bias = &dt_bias,
        .in_a = &in_a, .in_b = &in_b, .qkv = &qkv, .z = &z_weight,
        .norm = &norm, .out = &out_weight,
    };
    ds4_model compact;
    memset(&compact, 0, sizeof(compact));
    compact.fd = -1;
    compact.map = fixture;
    compact.size = fixture_bytes;
    printf("compact real GDN fixture: %.3f MiB (backbone mmap %.3f GiB)\n",
           (double)fixture_bytes / 1048576.0,
           (double)model.size / 1073741824.0);

    const uint64_t input_count = (uint64_t)TEST_ROWS * TEST_HIDDEN;
    const uint64_t qkv_count = (uint64_t)TEST_ROWS * TEST_CONV_DIM;
    const uint64_t value_count = (uint64_t)TEST_ROWS * TEST_VALUE_DIM;
    const uint64_t control_count =
        (uint64_t)TEST_ROWS * TEST_VALUE_HEADS;
    const uint64_t conv_state_count =
        (uint64_t)TEST_CONV_DIM * TEST_CONV_KERNEL;
    const uint64_t recurrent_count =
        (uint64_t)TEST_VALUE_HEADS * TEST_HEAD_DIM * TEST_HEAD_DIM;
    float *input = malloc(input_count * sizeof(float));
    float *cpu_qkv = malloc(qkv_count * sizeof(float));
    float *cpu_z = malloc(value_count * sizeof(float));
    float *cpu_a = malloc(control_count * sizeof(float));
    float *cpu_b = malloc(control_count * sizeof(float));
    float *gpu_qkv_raw = malloc(qkv_count * sizeof(float));
    float *gpu_qkv = malloc(qkv_count * sizeof(float));
    float *gpu_z = malloc(value_count * sizeof(float));
    float *gpu_a = malloc(control_count * sizeof(float));
    float *gpu_b = malloc(control_count * sizeof(float));
    float *gpu_beta = malloc(control_count * sizeof(float));
    float *gpu_g = malloc(control_count * sizeof(float));
    float *gpu_core = malloc(value_count * sizeof(float));
    float *gpu_gated = malloc(value_count * sizeof(float));
    float *gpu_output = malloc(input_count * sizeof(float));
    float *chunk_output = malloc(input_count * sizeof(float));
    float *ref_qkv = malloc(qkv_count * sizeof(float));
    float *ref_beta = malloc(control_count * sizeof(float));
    float *ref_g = malloc(control_count * sizeof(float));
    float *ref_core = malloc(value_count * sizeof(float));
    float *ref_gated = malloc(value_count * sizeof(float));
    float *ref_output = malloc(input_count * sizeof(float));
    float *ref_conv_state = calloc(conv_state_count, sizeof(float));
    float *ref_recurrent = calloc(recurrent_count, sizeof(float));
    float *gpu_conv_state = malloc(conv_state_count * sizeof(float));
    float *gpu_recurrent = malloc(recurrent_count * sizeof(float));
    REQUIRE(input && cpu_qkv && cpu_z && cpu_a && cpu_b && gpu_qkv_raw &&
            gpu_qkv && gpu_z && gpu_a && gpu_b && gpu_beta && gpu_g &&
            gpu_core && gpu_gated && gpu_output && chunk_output && ref_qkv &&
            ref_beta && ref_g && ref_core && ref_gated && ref_output &&
            ref_conv_state && ref_recurrent && gpu_conv_state &&
            gpu_recurrent, "real GDN host allocations");
    for (uint64_t i = 0; i < input_count; i++)
        input[i] = 0.43f * sinf((float)(i + 3u) * 0.0087f) +
                   0.17f * cosf((float)(i + 31u) * 0.019f) +
                   (float)((int)(i % 29u) - 14) * 0.0017f;
    gdn_matvec_rows(cpu_qkv, &compact, &qkv, input, TEST_ROWS,
                    TEST_HIDDEN, TEST_CONV_DIM);
    gdn_matvec_rows(cpu_z, &compact, &z_weight, input, TEST_ROWS,
                    TEST_HIDDEN, TEST_VALUE_DIM);
    gdn_matvec_rows(cpu_a, &compact, &in_a, input, TEST_ROWS,
                    TEST_HIDDEN, TEST_VALUE_HEADS);
    gdn_matvec_rows(cpu_b, &compact, &in_b, input, TEST_ROWS,
                    TEST_HIDDEN, TEST_VALUE_HEADS);

    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(unsetenv("DS4_CUDA_COPY_MODEL") == 0,
            "disable whole-fixture copy override");
    REQUIRE(ds4_gpu_set_model_map(fixture, fixture_bytes),
            "register compact real GDN fixture");
    ds4_qwen_gdn_ws ws;
    ds4_qwen_gdn_state state;
    REQUIRE(qwen4exp_gdn_ws_alloc(
                &ws, TEST_ROWS, TEST_HIDDEN, TEST_KEY_HEADS,
                TEST_VALUE_HEADS, TEST_HEAD_DIM),
            "allocate real GDN workspace");
    REQUIRE(qwen4exp_gdn_state_alloc(
                &state, TEST_CONV_DIM, TEST_CONV_KERNEL,
                TEST_VALUE_HEADS, TEST_HEAD_DIM),
            "allocate real GDN persistent state");
    ds4_gpu_tensor *dinput =
        ds4_gpu_tensor_alloc(input_count * sizeof(float));
    ds4_gpu_tensor *doutput =
        ds4_gpu_tensor_alloc(input_count * sizeof(float));
    REQUIRE(dinput && doutput, "real GDN device IO allocation");
    REQUIRE(ds4_gpu_tensor_write(
                dinput, 0, input, input_count * sizeof(float)),
            "real GDN input upload");
    REQUIRE(qwen4exp_gdn_forward(
                &ws, &state, &compact, &weights,
                dinput, doutput, TEST_ROWS),
            "complete real GDN forward");
    REQUIRE(ds4_gpu_tensor_read(ws.qkv_raw, 0, gpu_qkv_raw,
                                qkv_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.qkv, 0, gpu_qkv,
                                qkv_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.z, 0, gpu_z,
                                value_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.a, 0, gpu_a,
                                control_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.b, 0, gpu_b,
                                control_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.beta, 0, gpu_beta,
                                control_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.g, 0, gpu_g,
                                control_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.core, 0, gpu_core,
                                value_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.gated, 0, gpu_gated,
                                value_count * sizeof(float)) &&
            ds4_gpu_tensor_read(doutput, 0, gpu_output,
                                input_count * sizeof(float)) &&
            ds4_gpu_tensor_read(state.conv, 0, gpu_conv_state,
                                conv_state_count * sizeof(float)) &&
            ds4_gpu_tensor_read(state.recurrent, 0, gpu_recurrent,
                                recurrent_count * sizeof(float)),
            "real GDN stage readback");

    gdn_compare("real Q8_0 QKV projection", gpu_qkv_raw, cpu_qkv,
                qkv_count, 1.5e-2, 2.0e-4, 0.08f);
    gdn_compare("real Q8_0 z projection", gpu_z, cpu_z,
                value_count, 1.5e-2, 2.0e-4, 0.08f);
    gdn_compare("real Q8_0 a projection", gpu_a, cpu_a,
                control_count, 1.5e-2, 2.0e-4, 0.08f);
    gdn_compare("real Q8_0 b projection", gpu_b, cpu_b,
                control_count, 1.5e-2, 2.0e-4, 0.08f);

    const uint16_t *conv_weight =
        (const uint16_t *)tensor_data(&compact, &conv);
    const float *a_log_values =
        (const float *)tensor_data(&compact, &a_log);
    const float *dt_bias_values =
        (const float *)tensor_data(&compact, &dt_bias);
    const float *norm_values =
        (const float *)tensor_data(&compact, &norm);
    gdn_conv_reference(ref_qkv, ref_conv_state, gpu_qkv_raw,
                       conv_weight, TEST_ROWS);
    gdn_controls_reference(ref_beta, ref_g, gpu_b, gpu_a,
                           a_log_values, dt_bias_values, TEST_ROWS);
    gdn_recurrent_reference(ref_core, ref_recurrent, gpu_qkv,
                            gpu_beta, gpu_g, TEST_ROWS);
    gdn_norm_reference(ref_gated, gpu_core, gpu_z, norm_values, TEST_ROWS);
    gdn_matvec_rows(ref_output, &compact, &out_weight, gpu_gated,
                    TEST_ROWS, TEST_VALUE_DIM, TEST_HIDDEN);
    gdn_compare("real BF16 depthwise conv + SiLU", gpu_qkv, ref_qkv,
                qkv_count, 3.0e-6, 3.0e-6, 3.0e-6f);
    gdn_compare("real sigmoid/decay controls", gpu_beta, ref_beta,
                control_count, 3.0e-6, 3.0e-6, 3.0e-6f);
    gdn_compare("real recurrent Gated Delta core", gpu_core, ref_core,
                value_count, 3.0e-6, 3.0e-6, 3.0e-6f);
    gdn_compare("real gated per-head RMSNorm", gpu_gated, ref_gated,
                value_count, 3.0e-6, 3.0e-6, 3.0e-6f);
    gdn_compare("real Q8_0 output projection", gpu_output, ref_output,
                input_count, 1.5e-2, 2.0e-4, 0.08f);
    gdn_compare("real four-slot convolution final state", gpu_conv_state,
                ref_conv_state, conv_state_count, 0.0, 0.0, 0.0f);
    gdn_compare("real recurrent final state", gpu_recurrent,
                ref_recurrent, recurrent_count, 3.0e-6, 3.0e-6, 3.0e-6f);

    /* End-to-end CPU-dequantized reference starts from its own projections. */
    memset(ref_conv_state, 0, conv_state_count * sizeof(float));
    memset(ref_recurrent, 0, recurrent_count * sizeof(float));
    gdn_conv_reference(ref_qkv, ref_conv_state, cpu_qkv,
                       conv_weight, TEST_ROWS);
    gdn_controls_reference(ref_beta, ref_g, cpu_b, cpu_a,
                           a_log_values, dt_bias_values, TEST_ROWS);
    gdn_recurrent_reference(ref_core, ref_recurrent, ref_qkv,
                            ref_beta, ref_g, TEST_ROWS);
    gdn_norm_reference(ref_gated, ref_core, cpu_z, norm_values, TEST_ROWS);
    gdn_matvec_rows(ref_output, &compact, &out_weight, ref_gated,
                    TEST_ROWS, TEST_VALUE_DIM, TEST_HIDDEN);
    gdn_compare("real complete GDN vs CPU dequant reference", gpu_output,
                ref_output, input_count, 3.0e-2, 5.0e-4, 0.12f);

    const uint32_t chunks[] = {2u, 1u};
    REQUIRE(qwen4exp_gdn_state_reset(&state), "real GDN chunk state reset");
    gdn_run_chunked(&ws, &state, &compact, &weights,
                    dinput, doutput, chunks, 2u);
    REQUIRE(ds4_gpu_tensor_read(
                doutput, 0, chunk_output, input_count * sizeof(float)),
            "real GDN chunk output readback");
    gdn_compare("real GDN arbitrary-chunk parity", chunk_output, gpu_output,
                input_count, 1.0e-7, 1.0e-7, 1.0e-6f);
    const uint32_t decode[] = {1u, 1u, 1u};
    REQUIRE(qwen4exp_gdn_state_reset(&state), "real GDN decode state reset");
    gdn_run_chunked(&ws, &state, &compact, &weights,
                    dinput, doutput, decode, 3u);
    REQUIRE(ds4_gpu_tensor_read(
                doutput, 0, chunk_output, input_count * sizeof(float)),
            "real GDN decode output readback");
    gdn_compare("real GDN one-token decode parity", chunk_output, gpu_output,
                input_count, 1.0e-7, 1.0e-7, 1.0e-6f);

    struct rusage usage;
    memset(&usage, 0, sizeof(usage));
    REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0, "read real GDN RSS");
    printf("real GDN: scratch=%.3f MiB state=%.3f MiB fixture=%.3f MiB "
           "peak-RSS=%ld KiB\n",
           (double)ws.bytes / 1048576.0, (double)state.bytes / 1048576.0,
           (double)fixture_bytes / 1048576.0, usage.ru_maxrss);

    REQUIRE(ds4_gpu_synchronize(), "final real GDN synchronization");
    ds4_gpu_tensor_free(doutput);
    ds4_gpu_tensor_free(dinput);
    qwen4exp_gdn_state_free(&state);
    qwen4exp_gdn_ws_free(&ws);
    ds4_gpu_unregister_model_map(fixture);
    ds4_gpu_cleanup();
    free(gpu_recurrent); free(gpu_conv_state);
    free(ref_recurrent); free(ref_conv_state); free(ref_output);
    free(ref_gated); free(ref_core); free(ref_g); free(ref_beta); free(ref_qkv);
    free(chunk_output); free(gpu_output); free(gpu_gated); free(gpu_core);
    free(gpu_g); free(gpu_beta); free(gpu_b); free(gpu_a); free(gpu_z);
    free(gpu_qkv); free(gpu_qkv_raw); free(cpu_b); free(cpu_a);
    free(cpu_z); free(cpu_qkv); free(input);
    free(fixture);
    ds4_threads_shutdown();
    model_close(&model);
    puts("all real Qwen3.8 Gated DeltaNet checks passed");
    return 0;
}
