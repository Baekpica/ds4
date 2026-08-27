/* Real-artifact CUDA parity for the complete Qwen4Exp text MoE body.
 *
 * The public layer-2 router first chooses the real top-10 experts. Only those
 * expert slices plus the always-active shared matrices are copied into a
 * compact fixture, keeping the published backbone mmap non-resident while
 * preserving the exact published variant-selected bytes used by the forward. */
#include "../ds4.c"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

enum {
    TEST_LAYER = 2,
    TEST_HIDDEN = 2560,
    TEST_EXPERT_FF = 640,
    TEST_MAIN = 512,
    TEST_TAIL = 128,
    TEST_FULL_EXPERTS = 512,
    TEST_USED = 10,
};

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint64_t moe_align(uint64_t value) {
    return (value + 4095u) & ~UINT64_C(4095);
}

static uint64_t moe_tensor_row_bytes(const ds4_tensor *tensor) {
    REQUIRE(tensor && tensor->ndim >= 2u && tensor->dim[1] != 0u,
            "tensor has rows");
    uint64_t rows = tensor->dim[1];
    if (tensor->ndim == 3u) rows *= tensor->dim[2];
    REQUIRE(rows != 0u && tensor->bytes % rows == 0u,
            "tensor rows divide byte size");
    return tensor->bytes / rows;
}

static uint64_t moe_fixture_reserve(uint64_t cursor, uint64_t bytes) {
    return moe_align(cursor) + bytes;
}

static ds4_tensor moe_copy_whole(
        unsigned char *fixture,
        uint64_t fixture_bytes,
        uint64_t *cursor,
        const ds4_model *model,
        const ds4_tensor *source) {
    const uint64_t offset = moe_align(*cursor);
    REQUIRE(offset <= fixture_bytes && source->bytes <= fixture_bytes - offset,
            "whole fixture tensor in range");
    memcpy(fixture + offset, tensor_data(model, source), (size_t)source->bytes);
    ds4_tensor copy = *source;
    copy.rel_offset = offset;
    copy.abs_offset = offset;
    *cursor = offset + source->bytes;
    return copy;
}

static ds4_tensor moe_copy_router_rows(
        unsigned char *fixture,
        uint64_t fixture_bytes,
        uint64_t *cursor,
        const ds4_model *model,
        const ds4_tensor *source,
        const int32_t *experts,
        uint32_t count) {
    REQUIRE(source->ndim == 2u && source->dim[1] == TEST_FULL_EXPERTS,
            "source router layout");
    const uint64_t row_bytes = moe_tensor_row_bytes(source);
    const uint64_t bytes = row_bytes * count;
    const uint64_t offset = moe_align(*cursor);
    REQUIRE(offset <= fixture_bytes && bytes <= fixture_bytes - offset,
            "router fixture tensor in range");
    const unsigned char *data = (const unsigned char *)tensor_data(model, source);
    for (uint32_t compact = 0; compact < count; compact++) {
        REQUIRE(experts[compact] >= 0 && experts[compact] < TEST_FULL_EXPERTS,
                "selected router expert in range");
        memcpy(fixture + offset + (uint64_t)compact * row_bytes,
               data + (uint64_t)experts[compact] * row_bytes,
               (size_t)row_bytes);
    }
    ds4_tensor copy = *source;
    copy.dim[1] = count;
    copy.elements = copy.dim[0] * count;
    copy.bytes = bytes;
    copy.rel_offset = offset;
    copy.abs_offset = offset;
    *cursor = offset + bytes;
    return copy;
}

static ds4_tensor moe_copy_expert_slices(
        unsigned char *fixture,
        uint64_t fixture_bytes,
        uint64_t *cursor,
        const ds4_model *model,
        const ds4_tensor *source,
        const int32_t *experts,
        uint32_t count) {
    REQUIRE(source->ndim == 3u && source->dim[2] == TEST_FULL_EXPERTS,
            "source expert layout");
    REQUIRE(source->bytes % source->dim[2] == 0u,
            "expert slices divide byte size");
    const uint64_t expert_bytes = source->bytes / source->dim[2];
    const uint64_t bytes = expert_bytes * count;
    const uint64_t offset = moe_align(*cursor);
    REQUIRE(offset <= fixture_bytes && bytes <= fixture_bytes - offset,
            "expert fixture tensor in range");
    const unsigned char *data = (const unsigned char *)tensor_data(model, source);
    for (uint32_t compact = 0; compact < count; compact++) {
        REQUIRE(experts[compact] >= 0 && experts[compact] < TEST_FULL_EXPERTS,
                "selected expert slice in range");
        memcpy(fixture + offset + (uint64_t)compact * expert_bytes,
               data + (uint64_t)experts[compact] * expert_bytes,
               (size_t)expert_bytes);
    }
    ds4_tensor copy = *source;
    copy.dim[2] = count;
    copy.elements = copy.dim[0] * copy.dim[1] * count;
    copy.bytes = bytes;
    copy.rel_offset = offset;
    copy.abs_offset = offset;
    *cursor = offset + bytes;
    return copy;
}

static void qwen_router_reference(
        int32_t *selected,
        float *weights,
        const float *logits) {
    float score[TEST_FULL_EXPERTS];
    memcpy(score, logits, sizeof(score));
    float max_selected = -INFINITY;
    for (uint32_t slot = 0; slot < TEST_USED; slot++) {
        int32_t best = -1;
        float best_score = -INFINITY;
        for (uint32_t expert = 0; expert < TEST_FULL_EXPERTS; expert++) {
            if (score[expert] > best_score) {
                best_score = score[expert];
                best = (int32_t)expert;
            }
        }
        selected[slot] = best;
        weights[slot] = best_score;
        if (best_score > max_selected) max_selected = best_score;
        score[(uint32_t)best] = -INFINITY;
    }
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < TEST_USED; slot++) {
        weights[slot] = expf(weights[slot] - max_selected);
        sum += weights[slot];
    }
    for (uint32_t slot = 0; slot < TEST_USED; slot++) weights[slot] /= sum;
}

static float q5_0_tail_row_dot(
        const ds4_model *model,
        const ds4_tensor *tensor,
        uint32_t expert,
        uint32_t out_row,
        const float *input) {
    REQUIRE(tensor->type == DS4_TENSOR_Q5_0 && tensor->dim[0] == TEST_TAIL,
            "Q5_0 tail tensor contract");
    const uint64_t row_bytes = moe_tensor_row_bytes(tensor);
    REQUIRE(row_bytes == 4u * 22u, "Q5_0 tail row byte count");
    const uint8_t *row = (const uint8_t *)tensor_data(model, tensor) +
        ((uint64_t)expert * TEST_HIDDEN + out_row) * row_bytes;
    float sum = 0.0f;
    for (uint32_t block = 0; block < 4u; block++) {
        const uint8_t *p = row + block * 22u;
        uint16_t scale_bits;
        memcpy(&scale_bits, p, sizeof(scale_bits));
        const float scale = f16_to_f32(scale_bits);
        const uint32_t qh = (uint32_t)p[2] |
                            ((uint32_t)p[3] << 8u) |
                            ((uint32_t)p[4] << 16u) |
                            ((uint32_t)p[5] << 24u);
        const uint8_t *qs = p + 6u;
        for (uint32_t lane = 0; lane < 16u; lane++) {
            const uint32_t q0 = (qs[lane] & 15u) |
                                (((qh >> lane) & 1u) << 4u);
            const uint32_t q1 = (qs[lane] >> 4u) |
                                (((qh >> (lane + 16u)) & 1u) << 4u);
            const uint32_t base = block * 32u;
            sum += input[base + lane] * scale * ((float)q0 - 16.0f);
            sum += input[base + lane + 16u] * scale * ((float)q1 - 16.0f);
        }
    }
    return sum;
}

static void compare_metrics(const char *name, const float *got,
                            const float *want, uint64_t count,
                            double max_rel_rms, double max_one_minus_cos) {
    double error2 = 0.0;
    double ref2 = 0.0;
    double got2 = 0.0;
    double dot = 0.0;
    float max_abs = 0.0f;
    uint64_t max_i = 0u;
    for (uint64_t i = 0; i < count; i++) {
        const double error = (double)got[i] - want[i];
        error2 += error * error;
        ref2 += (double)want[i] * want[i];
        got2 += (double)got[i] * got[i];
        dot += (double)got[i] * want[i];
        const float abs_error = fabsf(got[i] - want[i]);
        if (abs_error > max_abs) {
            max_abs = abs_error;
            max_i = i;
        }
    }
    const double rel_rms = ref2 > 0.0 ? sqrt(error2 / ref2) : sqrt(error2);
    const double one_minus_cos = ref2 > 0.0 && got2 > 0.0
        ? 1.0 - dot / sqrt(got2 * ref2) : 0.0;
    printf("%-48s rel_rms %.4e 1-cos %.4e max %.5g @%llu\n",
           name, rel_rms, one_minus_cos, max_abs,
           (unsigned long long)max_i);
    REQUIRE(rel_rms <= max_rel_rms, "real MoE relative RMS gate");
    REQUIRE(one_minus_cos <= max_one_minus_cos, "real MoE cosine gate");
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
    ds4_weights weights;
    weights_bind(&weights, &model, false, 0, DS4_N_LAYER - 1u, true, false);
    const ds4_layer_weights *layer = &weights.layer[TEST_LAYER];
    ds4_str quantization = {0};
    uint32_t gate_type = 0u, down_type = 0u, tail_type = 0u;
    REQUIRE(model_get_string(&model, "general.quantization", &quantization) &&
            qwen4exp_ssd_precision_types(
                quantization, NULL, &gate_type, &down_type, &tail_type),
            "published SSD-PLE precision map");
    REQUIRE(layer->ffn_gate_inp->type == DS4_TENSOR_F32 &&
            layer->ffn_gate_exps->type == gate_type &&
            layer->ffn_up_exps->type == gate_type &&
            layer->ffn_down_exps->type == down_type &&
            layer->ffn_down_exps_tail->type == tail_type &&
            layer->ffn_gate_shexp->type == DS4_TENSOR_Q8_0 &&
            layer->ffn_up_shexp->type == DS4_TENSOR_Q8_0 &&
            layer->ffn_down_shexp->type == DS4_TENSOR_Q8_0 &&
            layer->ffn_shexp_gate_inp->type == DS4_TENSOR_F32,
            "published interior-layer recipe types");

    float *input = (float *)malloc(TEST_HIDDEN * sizeof(*input));
    float *router_logits = (float *)malloc(TEST_FULL_EXPERTS * sizeof(*router_logits));
    float *cpu_mid = (float *)malloc(
        (uint64_t)TEST_USED * TEST_EXPERT_FF * sizeof(*cpu_mid));
    float *cpu_down = (float *)malloc(
        (uint64_t)TEST_USED * TEST_HIDDEN * sizeof(*cpu_down));
    float *cpu_output = (float *)calloc(TEST_HIDDEN, sizeof(*cpu_output));
    float *cpu_shared_gate = (float *)malloc(TEST_EXPERT_FF * sizeof(float));
    float *cpu_shared_up = (float *)malloc(TEST_EXPERT_FF * sizeof(float));
    float *cpu_shared_mid = (float *)malloc(TEST_EXPERT_FF * sizeof(float));
    float *cpu_shared_out = (float *)malloc(TEST_HIDDEN * sizeof(float));
    float *tmp_gate = (float *)malloc(TEST_EXPERT_FF * sizeof(float));
    float *tmp_up = (float *)malloc(TEST_EXPERT_FF * sizeof(float));
    float *tmp_main = (float *)malloc(TEST_HIDDEN * sizeof(float));
    REQUIRE(input && router_logits && cpu_mid && cpu_down && cpu_output &&
            cpu_shared_gate && cpu_shared_up && cpu_shared_mid &&
            cpu_shared_out && tmp_gate && tmp_up && tmp_main,
            "real MoE CPU allocation");
    for (uint32_t i = 0; i < TEST_HIDDEN; i++) {
        input[i] = 0.47f * sinf((float)(i + 3u) * 0.0091f) +
                   0.16f * cosf((float)(i + 17u) * 0.021f) +
                   (float)((int)(i % 23u) - 11) * 0.002f;
    }

    int32_t selected[TEST_USED];
    float router_weights[TEST_USED];
    matvec_f32(router_logits, &model, layer->ffn_gate_inp, input);
    qwen_router_reference(selected, router_weights, router_logits);
    printf("real layer-%u experts:", TEST_LAYER);
    for (uint32_t slot = 0; slot < TEST_USED; slot++)
        printf(" %d(%.6f)", selected[slot], router_weights[slot]);
    putchar('\n');

    for (uint32_t slot = 0; slot < TEST_USED; slot++) {
        const uint32_t expert = (uint32_t)selected[slot];
        exaone_linear_expert(tmp_gate, &model, layer->ffn_gate_exps,
                             expert, input);
        exaone_linear_expert(tmp_up, &model, layer->ffn_up_exps,
                             expert, input);
        float *mid = cpu_mid + (uint64_t)slot * TEST_EXPERT_FF;
        for (uint32_t i = 0; i < TEST_EXPERT_FF; i++) {
            mid[i] = tmp_gate[i] / (1.0f + expf(-tmp_gate[i])) *
                     tmp_up[i] * router_weights[slot];
        }
        exaone_linear_expert(tmp_main, &model, layer->ffn_down_exps,
                             expert, mid);
        float *down = cpu_down + (uint64_t)slot * TEST_HIDDEN;
        for (uint32_t out = 0; out < TEST_HIDDEN; out++) {
            down[out] = tmp_main[out] + q5_0_tail_row_dot(
                &model, layer->ffn_down_exps_tail, expert, out,
                mid + TEST_MAIN);
            cpu_output[out] += down[out];
        }
    }

    matvec_q8_0(cpu_shared_gate, &model, layer->ffn_gate_shexp, input);
    matvec_q8_0(cpu_shared_up, &model, layer->ffn_up_shexp, input);
    for (uint32_t i = 0; i < TEST_EXPERT_FF; i++) {
        cpu_shared_mid[i] = cpu_shared_gate[i] /
            (1.0f + expf(-cpu_shared_gate[i])) * cpu_shared_up[i];
    }
    matvec_q8_0(cpu_shared_out, &model, layer->ffn_down_shexp,
                cpu_shared_mid);
    float shared_logit;
    matvec_f32(&shared_logit, &model, layer->ffn_shexp_gate_inp, input);
    const float shared_scale = 1.0f / (1.0f + expf(-shared_logit));
    for (uint32_t i = 0; i < TEST_HIDDEN; i++) {
        cpu_shared_out[i] *= shared_scale;
        cpu_output[i] += cpu_shared_out[i];
    }

    uint64_t fixture_bytes = 4096u;
    fixture_bytes = moe_fixture_reserve(
        fixture_bytes,
        moe_tensor_row_bytes(layer->ffn_gate_inp) * TEST_USED);
    const ds4_tensor *expert_sources[] = {
        layer->ffn_gate_exps, layer->ffn_up_exps,
        layer->ffn_down_exps, layer->ffn_down_exps_tail,
    };
    for (size_t i = 0; i < sizeof(expert_sources) / sizeof(expert_sources[0]); i++)
        fixture_bytes = moe_fixture_reserve(
            fixture_bytes,
            expert_sources[i]->bytes / TEST_FULL_EXPERTS * TEST_USED);
    const ds4_tensor *whole_sources[] = {
        layer->ffn_gate_shexp, layer->ffn_up_shexp,
        layer->ffn_down_shexp, layer->ffn_shexp_gate_inp,
    };
    for (size_t i = 0; i < sizeof(whole_sources) / sizeof(whole_sources[0]); i++)
        fixture_bytes = moe_fixture_reserve(fixture_bytes, whole_sources[i]->bytes);
    fixture_bytes = moe_align(fixture_bytes);

    unsigned char *fixture = NULL;
    REQUIRE(posix_memalign((void **)&fixture, 4096u,
                           (size_t)fixture_bytes) == 0,
            "compact real MoE fixture allocation");
    memset(fixture, 0, (size_t)fixture_bytes);
    uint64_t cursor = 4096u;
    ds4_tensor router = moe_copy_router_rows(
        fixture, fixture_bytes, &cursor, &model,
        layer->ffn_gate_inp, selected, TEST_USED);
    ds4_tensor gate_exps = moe_copy_expert_slices(
        fixture, fixture_bytes, &cursor, &model,
        layer->ffn_gate_exps, selected, TEST_USED);
    ds4_tensor up_exps = moe_copy_expert_slices(
        fixture, fixture_bytes, &cursor, &model,
        layer->ffn_up_exps, selected, TEST_USED);
    ds4_tensor down_main = moe_copy_expert_slices(
        fixture, fixture_bytes, &cursor, &model,
        layer->ffn_down_exps, selected, TEST_USED);
    ds4_tensor down_tail = moe_copy_expert_slices(
        fixture, fixture_bytes, &cursor, &model,
        layer->ffn_down_exps_tail, selected, TEST_USED);
    ds4_tensor shared_gate = moe_copy_whole(
        fixture, fixture_bytes, &cursor, &model, layer->ffn_gate_shexp);
    ds4_tensor shared_up = moe_copy_whole(
        fixture, fixture_bytes, &cursor, &model, layer->ffn_up_shexp);
    ds4_tensor shared_down = moe_copy_whole(
        fixture, fixture_bytes, &cursor, &model, layer->ffn_down_shexp);
    ds4_tensor shared_gate_inp = moe_copy_whole(
        fixture, fixture_bytes, &cursor, &model, layer->ffn_shexp_gate_inp);
    REQUIRE(moe_align(cursor) == fixture_bytes, "fixture byte census exact");

    ds4_layer_weights compact_layer = *layer;
    compact_layer.ffn_gate_inp = &router;
    compact_layer.ffn_gate_exps = &gate_exps;
    compact_layer.ffn_up_exps = &up_exps;
    compact_layer.ffn_down_exps = &down_main;
    compact_layer.ffn_down_exps_tail = &down_tail;
    compact_layer.ffn_gate_shexp = &shared_gate;
    compact_layer.ffn_up_shexp = &shared_up;
    compact_layer.ffn_down_shexp = &shared_down;
    compact_layer.ffn_shexp_gate_inp = &shared_gate_inp;
    ds4_model fixture_model;
    memset(&fixture_model, 0, sizeof(fixture_model));
    fixture_model.fd = -1;
    fixture_model.map = fixture;
    fixture_model.size = fixture_bytes;
    printf("compact selected-expert fixture: %.3f MiB "
           "(backbone mmap %.3f GiB)\n",
           (double)fixture_bytes / 1048576.0,
           (double)model.size / 1073741824.0);

    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(unsetenv("DS4_CUDA_COPY_MODEL") == 0,
            "disable whole-fixture copy override");
    REQUIRE(setenv("DS4_QWEN_Q5_TAIL_EXPERT_MAJOR", "1", 1) == 0,
            "force expert-major Q5_0 tail real-artifact path");
    REQUIRE(ds4_gpu_set_model_map(fixture, fixture_bytes),
            "register compact real MoE fixture");
    ds4_qwen_moe_ws ws;
    REQUIRE(qwen4exp_moe_ws_alloc(
                &ws, 1u, TEST_USED, TEST_USED,
                TEST_HIDDEN, TEST_EXPERT_FF, TEST_EXPERT_FF),
            "allocate Qwen real MoE workspace");
    ds4_gpu_tensor *d_input = ds4_gpu_tensor_alloc(
        TEST_HIDDEN * sizeof(float));
    ds4_gpu_tensor *d_output = ds4_gpu_tensor_alloc(
        TEST_HIDDEN * sizeof(float));
    REQUIRE(d_input && d_output, "real MoE device IO allocation");
    REQUIRE(ds4_gpu_tensor_write(
                d_input, 0, input, TEST_HIDDEN * sizeof(float)),
            "real MoE input upload");
    REQUIRE(qwen4exp_moe_forward(
                &ws, &fixture_model, &compact_layer,
                d_input, d_output, 1u),
            "complete real Qwen MoE forward");

    int32_t gpu_ids[TEST_USED];
    float gpu_weights[TEST_USED];
    float *gpu_mid = (float *)malloc(
        (uint64_t)TEST_USED * TEST_EXPERT_FF * sizeof(*gpu_mid));
    float *gpu_down = (float *)malloc(
        (uint64_t)TEST_USED * TEST_HIDDEN * sizeof(*gpu_down));
    float *gpu_shared = (float *)malloc(TEST_HIDDEN * sizeof(*gpu_shared));
    float *gpu_output = (float *)malloc(TEST_HIDDEN * sizeof(*gpu_output));
    REQUIRE(gpu_mid && gpu_down && gpu_shared && gpu_output,
            "real MoE GPU readback allocation");
    REQUIRE(ds4_gpu_tensor_read(ws.selected, 0, gpu_ids, sizeof(gpu_ids)),
            "real MoE selected ids download");
    REQUIRE(ds4_gpu_tensor_read(ws.router_weights, 0,
                                gpu_weights, sizeof(gpu_weights)),
            "real MoE router weights download");
    REQUIRE(ds4_gpu_tensor_read(
                ws.routed_mid, 0, gpu_mid,
                (uint64_t)TEST_USED * TEST_EXPERT_FF * sizeof(float)),
            "real MoE routed mid download");
    REQUIRE(ds4_gpu_tensor_read(
                ws.routed_down, 0, gpu_down,
                (uint64_t)TEST_USED * TEST_HIDDEN * sizeof(float)),
            "real MoE routed down download");
    REQUIRE(ds4_gpu_tensor_read(
                ws.shared_out, 0, gpu_shared,
                TEST_HIDDEN * sizeof(float)),
            "real MoE shared output download");
    REQUIRE(ds4_gpu_tensor_read(
                d_output, 0, gpu_output,
                TEST_HIDDEN * sizeof(float)),
            "real MoE final output download");

    for (uint32_t slot = 0; slot < TEST_USED; slot++) {
        REQUIRE(gpu_ids[slot] >= 0 && gpu_ids[slot] < TEST_USED,
                "compact selected id in range");
        REQUIRE(selected[(uint32_t)gpu_ids[slot]] == selected[slot],
                "real top-10 expert order exact");
    }
    for (uint32_t slot = 0; slot < TEST_USED; slot++) {
        REQUIRE(fabsf(gpu_weights[slot] - router_weights[slot]) <= 3.0e-6f,
                "real router selected weight parity");
    }
    printf("%-48s pass (10 ids and weights)\n",
           "real F32 router -> normalized top-10");
    compare_metrics("real routed gate/up -> weighted SwiGLU",
                    gpu_mid, cpu_mid,
                    (uint64_t)TEST_USED * TEST_EXPERT_FF,
                    2.0e-2, 2.0e-3);
    compare_metrics("real routed main + Q5_0 tail per expert",
                    gpu_down, cpu_down,
                    (uint64_t)TEST_USED * TEST_HIDDEN,
                    3.0e-2, 3.0e-3);
    compare_metrics("real sigmoid-gated Q8_0 shared expert",
                    gpu_shared, cpu_shared_out, TEST_HIDDEN,
                    2.0e-2, 2.0e-3);
    compare_metrics("real complete Qwen text MoE output",
                    gpu_output, cpu_output, TEST_HIDDEN,
                    3.0e-2, 3.0e-3);

    struct rusage usage;
    memset(&usage, 0, sizeof(usage));
    REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0, "read process RSS");
    printf("real MoE: workspace=%.3f MiB fixture=%.3f MiB peak-RSS=%ld KiB\n",
           (double)ws.bytes / 1048576.0,
           (double)fixture_bytes / 1048576.0, usage.ru_maxrss);

    REQUIRE(ds4_gpu_synchronize(), "final real MoE synchronization");
    free(gpu_output);
    free(gpu_shared);
    free(gpu_down);
    free(gpu_mid);
    ds4_gpu_tensor_free(d_output);
    ds4_gpu_tensor_free(d_input);
    qwen4exp_moe_ws_free(&ws);
    ds4_gpu_unregister_model_map(fixture);
    ds4_gpu_cleanup();
    free(fixture);
    ds4_threads_shutdown();
    model_close(&model);

    free(tmp_main);
    free(tmp_up);
    free(tmp_gate);
    free(cpu_shared_out);
    free(cpu_shared_mid);
    free(cpu_shared_up);
    free(cpu_shared_gate);
    free(cpu_output);
    free(cpu_down);
    free(cpu_mid);
    free(router_logits);
    free(input);
    puts("all real Qwen3.8 MoE forward checks passed");
    return 0;
}
