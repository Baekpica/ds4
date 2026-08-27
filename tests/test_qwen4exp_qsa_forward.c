/* Real-artifact H200 parity for a complete Qwen3.8 QSA layer.  Layer 3's
 * nine published tensors are copied into a compact fixture so the 90.96 GiB
 * public backbone remains mmap-only while the production high-level path is
 * exercised with its actual Q8_0/F32 bytes. */
#include "../ds4.c"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

enum {
    TEST_LAYER = 3,
    TEST_ROWS = 5,
    TEST_CONTEXT = 8,
    TEST_HIDDEN = 2560,
    TEST_INDEX_HEADS = 4,
    TEST_INDEX_DIM = 128,
    TEST_INDEX_Q_DIM = TEST_INDEX_HEADS * TEST_INDEX_DIM,
    TEST_INDEX_QK_DIM = TEST_INDEX_Q_DIM + TEST_INDEX_DIM,
    TEST_HEADS = 24,
    TEST_KV_HEADS = 2,
    TEST_HEAD_DIM = 256,
    TEST_ROTARY_DIM = 64,
    TEST_Q_DIM = TEST_HEADS * TEST_HEAD_DIM,
    TEST_Q_PROJ_DIM = 2 * TEST_Q_DIM,
    TEST_KV_DIM = TEST_KV_HEADS * TEST_HEAD_DIM,
    TEST_RATIO = 4,
    TEST_BUDGET = 2048,
    TEST_SELECTED_CAP = TEST_BUDGET + TEST_RATIO - 1,
};

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint64_t qsa_align(uint64_t value) {
    return (value + 4095u) & ~UINT64_C(4095);
}

static ds4_tensor qsa_copy_tensor(
        unsigned char *fixture, uint64_t fixture_bytes, uint64_t *cursor,
        const ds4_model *model, const ds4_tensor *source) {
    const uint64_t offset = qsa_align(*cursor);
    REQUIRE(offset <= fixture_bytes && source->bytes <= fixture_bytes - offset,
            "QSA fixture tensor range");
    memcpy(fixture + offset, tensor_data(model, source), (size_t)source->bytes);
    ds4_tensor copy = *source;
    copy.rel_offset = offset;
    copy.abs_offset = offset;
    *cursor = offset + source->bytes;
    return copy;
}

static void qsa_compare(const char *name, const float *got,
                        const float *want, uint64_t count,
                        double max_rel_rms, double max_one_minus_cos,
                        float max_abs_limit) {
    double error2 = 0.0, ref2 = 0.0, got2 = 0.0, dot = 0.0;
    float max_abs = 0.0f;
    uint64_t worst = 0u;
    for (uint64_t i = 0; i < count; i++) {
        const double error = (double)got[i] - want[i];
        error2 += error * error;
        ref2 += (double)want[i] * want[i];
        got2 += (double)got[i] * got[i];
        dot += (double)got[i] * want[i];
        const float absolute = fabsf(got[i] - want[i]);
        if (absolute > max_abs) { max_abs = absolute; worst = i; }
    }
    const double rel_rms = sqrt(error2 / fmax(ref2, 1.0e-30));
    const double one_minus_cos = ref2 > 0.0 && got2 > 0.0
        ? 1.0 - dot / sqrt(ref2 * got2) : 0.0;
    printf("%-50s rel_rms %.4e 1-cos %.4e max %.5g @%llu\n",
           name, rel_rms, one_minus_cos, max_abs,
           (unsigned long long)worst);
    REQUIRE(rel_rms <= max_rel_rms, "real QSA relative RMS gate");
    REQUIRE(one_minus_cos <= max_one_minus_cos,
            "real QSA cosine gate");
    REQUIRE(max_abs <= max_abs_limit, "real QSA maximum error gate");
}

static void qsa_matvec_rows(float *out, const ds4_model *model,
                            const ds4_tensor *weight, const float *input,
                            uint32_t rows, uint32_t in_dim,
                            uint32_t out_dim) {
    REQUIRE(weight->type == DS4_TENSOR_Q8_0,
            "real QSA projection is Q8_0");
    for (uint32_t row = 0; row < rows; row++)
        matvec_q8_0(out + (uint64_t)row * out_dim, model, weight,
                    input + (uint64_t)row * in_dim);
}

static void qsa_shared_norm(float *x, const float *weight,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            float *v = x + ((uint64_t)row * heads + head) * head_dim;
            double sum = 0.0;
            for (uint32_t d = 0; d < head_dim; d++)
                sum += (double)v[d] * v[d];
            const float scale =
                1.0f / sqrtf((float)(sum / head_dim) + DS4_RMS_EPS);
            for (uint32_t d = 0; d < head_dim; d++)
                v[d] *= scale * (1.0f + weight[d]);
        }
    }
}

static void qsa_rope(float *x, uint32_t rows, uint32_t heads,
                     uint32_t head_dim, uint32_t pos0) {
    float inv_freq[TEST_ROTARY_DIM / 2u];
    for (uint32_t d = 0; d < TEST_ROTARY_DIM / 2u; d++)
        inv_freq[d] = 1.0f / powf(DS4_ROPE_FREQ_BASE,
            (2.0f * (float)d) / (float)TEST_ROTARY_DIM);
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            float *v = x + ((uint64_t)row * heads + head) * head_dim;
            for (uint32_t d = 0; d < TEST_ROTARY_DIM / 2u; d++) {
                const float theta = (float)(pos0 + row) * inv_freq[d];
                const float c = cosf(theta), s = sinf(theta);
                const float x0 = v[d];
                const float x1 = v[d + TEST_ROTARY_DIM / 2u];
                v[d] = x0 * c - x1 * s;
                v[d + TEST_ROTARY_DIM / 2u] = x1 * c + x0 * s;
            }
        }
    }
}

static void qsa_split_index(float *query, float *raw, const float *qk) {
    for (uint32_t row = 0; row < TEST_ROWS; row++) {
        memcpy(query + (uint64_t)row * TEST_INDEX_Q_DIM,
               qk + (uint64_t)row * TEST_INDEX_QK_DIM,
               TEST_INDEX_Q_DIM * sizeof(float));
        memcpy(raw + (uint64_t)row * TEST_INDEX_DIM,
               qk + (uint64_t)row * TEST_INDEX_QK_DIM + TEST_INDEX_Q_DIM,
               TEST_INDEX_DIM * sizeof(float));
    }
}

static void qsa_split_query(float *query, float *gate,
                            const float *projected) {
    for (uint32_t row = 0; row < TEST_ROWS; row++) {
        for (uint32_t head = 0; head < TEST_HEADS; head++) {
            const float *source = projected +
                ((uint64_t)row * TEST_HEADS + head) * 2u * TEST_HEAD_DIM;
            float *q = query +
                ((uint64_t)row * TEST_HEADS + head) * TEST_HEAD_DIM;
            float *g = gate +
                ((uint64_t)row * TEST_HEADS + head) * TEST_HEAD_DIM;
            memcpy(q, source, TEST_HEAD_DIM * sizeof(float));
            memcpy(g, source + TEST_HEAD_DIM,
                   TEST_HEAD_DIM * sizeof(float));
        }
    }
}

static void qsa_pool_first_block(float *pooled, const float *raw,
                                 const float *weight) {
    for (uint32_t d = 0; d < TEST_INDEX_DIM; d++) {
        float sum = 0.0f;
        for (uint32_t row = 0; row < TEST_RATIO; row++)
            sum += raw[(uint64_t)row * TEST_INDEX_DIM + d];
        pooled[d] = sum / (float)TEST_RATIO;
    }
    qsa_shared_norm(pooled, weight, 1u, 1u, TEST_INDEX_DIM);
    qsa_rope(pooled, 1u, 1u, TEST_INDEX_DIM, 0u);
}

static void qsa_attention(float *out, const float *query, const float *gate,
                          const float *key, const float *value) {
    float score[TEST_ROWS];
    for (uint32_t row = 0; row < TEST_ROWS; row++) {
        for (uint32_t head = 0; head < TEST_HEADS; head++) {
            const uint32_t kv_head = head / (TEST_HEADS / TEST_KV_HEADS);
            const float *q = query +
                ((uint64_t)row * TEST_HEADS + head) * TEST_HEAD_DIM;
            float maximum = -INFINITY;
            for (uint32_t token = 0; token <= row; token++) {
                const float *k = key +
                    ((uint64_t)token * TEST_KV_HEADS + kv_head) *
                    TEST_HEAD_DIM;
                float dot = 0.0f;
                for (uint32_t d = 0; d < TEST_HEAD_DIM; d++)
                    dot += q[d] * k[d];
                score[token] = dot / sqrtf((float)TEST_HEAD_DIM);
                if (score[token] > maximum) maximum = score[token];
            }
            float denominator = 0.0f;
            for (uint32_t token = 0; token <= row; token++) {
                score[token] = expf(score[token] - maximum);
                denominator += score[token];
            }
            for (uint32_t d = 0; d < TEST_HEAD_DIM; d++) {
                float sum = 0.0f;
                for (uint32_t token = 0; token <= row; token++) {
                    const float *v = value +
                        ((uint64_t)token * TEST_KV_HEADS + kv_head) *
                        TEST_HEAD_DIM;
                    sum += score[token] / denominator * v[d];
                }
                const uint64_t at =
                    ((uint64_t)row * TEST_HEADS + head) * TEST_HEAD_DIM + d;
                out[at] = sum / (1.0f + expf(-gate[at]));
            }
        }
    }
}

static void qsa_run_chunks(ds4_qwen_qsa_ws *ws,
                           ds4_qwen_qsa_state *state,
                           const ds4_model *model,
                           const ds4_qwen_qsa_weights *weights,
                           ds4_gpu_tensor *input, ds4_gpu_tensor *output,
                           const uint32_t *chunks, uint32_t count) {
    uint32_t row = 0u;
    for (uint32_t c = 0; c < count; c++) {
        const uint64_t offset =
            (uint64_t)row * TEST_HIDDEN * sizeof(float);
        const uint64_t bytes =
            (uint64_t)chunks[c] * TEST_HIDDEN * sizeof(float);
        ds4_gpu_tensor *iv = ds4_gpu_tensor_view(input, offset, bytes);
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(output, offset, bytes);
        REQUIRE(iv && ov, "real QSA chunk views");
        REQUIRE(qwen4exp_qsa_forward(
                    ws, state, model, weights, iv, ov, chunks[c], row),
                "real QSA chunk forward");
        ds4_gpu_tensor_free(ov);
        ds4_gpu_tensor_free(iv);
        row += chunks[c];
    }
    REQUIRE(row == TEST_ROWS && state->length == TEST_ROWS,
            "real QSA chunks cover input");
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
    const ds4_qwen_qsa_weights *source =
        &all_weights.layer[TEST_LAYER].qwen_qsa;
    REQUIRE(ds4_qwen4exp_layer_is_full_attention(TEST_LAYER),
            "layer 3 is QSA");
    REQUIRE(source->index_qk->type == DS4_TENSOR_Q8_0 &&
            source->q->type == DS4_TENSOR_Q8_0 &&
            source->k->type == DS4_TENSOR_Q8_0 &&
            source->v->type == DS4_TENSOR_Q8_0 &&
            source->output->type == DS4_TENSOR_Q8_0 &&
            source->index_q_norm->type == DS4_TENSOR_F32 &&
            source->index_k_norm->type == DS4_TENSOR_F32 &&
            source->q_norm->type == DS4_TENSOR_F32 &&
            source->k_norm->type == DS4_TENSOR_F32,
            "published QSA recipe types");

    const ds4_tensor *tensors[] = {
        source->index_qk, source->index_q_norm, source->index_k_norm,
        source->q, source->q_norm, source->k, source->k_norm,
        source->v, source->output,
    };
    uint64_t fixture_bytes = 4096u;
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++)
        fixture_bytes = qsa_align(fixture_bytes) + tensors[i]->bytes;
    fixture_bytes = qsa_align(fixture_bytes);
    unsigned char *fixture = NULL;
    REQUIRE(posix_memalign((void **)&fixture, 4096u,
                           (size_t)fixture_bytes) == 0,
            "real QSA compact fixture allocation");
    memset(fixture, 0, (size_t)fixture_bytes);
    uint64_t cursor = 4096u;
    ds4_tensor index_qk = qsa_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->index_qk);
    ds4_tensor index_q_norm = qsa_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->index_q_norm);
    ds4_tensor index_k_norm = qsa_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->index_k_norm);
    ds4_tensor q_weight = qsa_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->q);
    ds4_tensor q_norm = qsa_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->q_norm);
    ds4_tensor k_weight = qsa_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->k);
    ds4_tensor k_norm = qsa_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->k_norm);
    ds4_tensor v_weight = qsa_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->v);
    ds4_tensor output_weight = qsa_copy_tensor(
        fixture, fixture_bytes, &cursor, &model, source->output);
    REQUIRE(qsa_align(cursor) == fixture_bytes, "real QSA fixture census");
    ds4_qwen_qsa_weights weights = {
        .index_qk = &index_qk, .index_q_norm = &index_q_norm,
        .index_k_norm = &index_k_norm, .q = &q_weight,
        .q_norm = &q_norm, .k = &k_weight, .k_norm = &k_norm,
        .v = &v_weight, .output = &output_weight,
    };
    ds4_model compact;
    memset(&compact, 0, sizeof(compact));
    compact.fd = -1;
    compact.map = fixture;
    compact.size = fixture_bytes;
    printf("compact real QSA fixture: %.3f MiB (backbone mmap %.3f GiB)\n",
           (double)fixture_bytes / 1048576.0,
           (double)model.size / 1073741824.0);

    const uint64_t input_count = (uint64_t)TEST_ROWS * TEST_HIDDEN;
    const uint64_t index_qk_count =
        (uint64_t)TEST_ROWS * TEST_INDEX_QK_DIM;
    const uint64_t index_q_count =
        (uint64_t)TEST_ROWS * TEST_INDEX_Q_DIM;
    const uint64_t qproj_count =
        (uint64_t)TEST_ROWS * TEST_Q_PROJ_DIM;
    const uint64_t q_count = (uint64_t)TEST_ROWS * TEST_Q_DIM;
    const uint64_t kv_count = (uint64_t)TEST_ROWS * TEST_KV_DIM;
#define HOST_F32(name, count) float *name = malloc((count) * sizeof(float));   \
    REQUIRE(name, #name " allocation")
    HOST_F32(input, input_count);
    HOST_F32(cpu_index_qk, index_qk_count);
    HOST_F32(cpu_index_q, index_q_count);
    HOST_F32(cpu_raw, (uint64_t)TEST_ROWS * TEST_INDEX_DIM);
    HOST_F32(cpu_pool, TEST_INDEX_DIM);
    HOST_F32(cpu_qproj, qproj_count);
    HOST_F32(cpu_query, q_count);
    HOST_F32(cpu_gate, q_count);
    HOST_F32(cpu_key, kv_count);
    HOST_F32(cpu_value, kv_count);
    HOST_F32(cpu_attention, q_count);
    HOST_F32(cpu_output, input_count);
    HOST_F32(gpu_index_qk, index_qk_count);
    HOST_F32(gpu_index_q, index_q_count);
    HOST_F32(gpu_raw, (uint64_t)TEST_ROWS * TEST_INDEX_DIM);
    HOST_F32(gpu_pool, TEST_INDEX_DIM);
    HOST_F32(gpu_qproj, qproj_count);
    HOST_F32(gpu_query, q_count);
    HOST_F32(gpu_gate, q_count);
    HOST_F32(gpu_key, kv_count);
    HOST_F32(gpu_value, kv_count);
    HOST_F32(gpu_attention, q_count);
    HOST_F32(gpu_output, input_count);
    HOST_F32(chunk_output, input_count);
#undef HOST_F32
    int32_t selected[TEST_ROWS * TEST_SELECTED_CAP];
    uint32_t counts[TEST_ROWS];

    for (uint64_t i = 0; i < input_count; i++)
        input[i] = 0.39f * sinf((float)(i + 7u) * 0.0091f) +
                   0.16f * cosf((float)(i + 37u) * 0.0173f) +
                   (float)((int)(i % 31u) - 15) * 0.0013f;
    qsa_matvec_rows(cpu_index_qk, &compact, &index_qk, input,
                    TEST_ROWS, TEST_HIDDEN, TEST_INDEX_QK_DIM);
    qsa_split_index(cpu_index_q, cpu_raw, cpu_index_qk);
    qsa_shared_norm(cpu_index_q,
                    (const float *)tensor_data(&compact, &index_q_norm),
                    TEST_ROWS, TEST_INDEX_HEADS, TEST_INDEX_DIM);
    qsa_rope(cpu_index_q, TEST_ROWS, TEST_INDEX_HEADS,
             TEST_INDEX_DIM, 0u);
    qsa_pool_first_block(
        cpu_pool, cpu_raw,
        (const float *)tensor_data(&compact, &index_k_norm));
    qsa_matvec_rows(cpu_qproj, &compact, &q_weight, input,
                    TEST_ROWS, TEST_HIDDEN, TEST_Q_PROJ_DIM);
    qsa_split_query(cpu_query, cpu_gate, cpu_qproj);
    qsa_matvec_rows(cpu_key, &compact, &k_weight, input,
                    TEST_ROWS, TEST_HIDDEN, TEST_KV_DIM);
    qsa_matvec_rows(cpu_value, &compact, &v_weight, input,
                    TEST_ROWS, TEST_HIDDEN, TEST_KV_DIM);
    qsa_shared_norm(cpu_query,
                    (const float *)tensor_data(&compact, &q_norm),
                    TEST_ROWS, TEST_HEADS, TEST_HEAD_DIM);
    qsa_shared_norm(cpu_key,
                    (const float *)tensor_data(&compact, &k_norm),
                    TEST_ROWS, TEST_KV_HEADS, TEST_HEAD_DIM);
    qsa_rope(cpu_query, TEST_ROWS, TEST_HEADS, TEST_HEAD_DIM, 0u);
    qsa_rope(cpu_key, TEST_ROWS, TEST_KV_HEADS, TEST_HEAD_DIM, 0u);
    qsa_attention(cpu_attention, cpu_query, cpu_gate, cpu_key, cpu_value);
    qsa_matvec_rows(cpu_output, &compact, &output_weight, cpu_attention,
                    TEST_ROWS, TEST_Q_DIM, TEST_HIDDEN);

    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(unsetenv("DS4_CUDA_COPY_MODEL") == 0,
            "disable QSA fixture whole copy");
    REQUIRE(ds4_gpu_set_model_map(fixture, fixture_bytes),
            "register compact real QSA fixture");
    ds4_qwen_qsa_ws ws;
    ds4_qwen_qsa_state state;
    REQUIRE(qwen4exp_qsa_ws_alloc(
                &ws, TEST_ROWS, TEST_CONTEXT, TEST_HIDDEN,
                TEST_INDEX_HEADS, TEST_INDEX_DIM, TEST_HEADS,
                TEST_KV_HEADS, TEST_HEAD_DIM, TEST_ROTARY_DIM,
                TEST_RATIO, TEST_BUDGET),
            "allocate real QSA workspace");
    REQUIRE(qwen4exp_qsa_state_alloc(
                &state, TEST_CONTEXT, TEST_RATIO, TEST_INDEX_DIM,
                TEST_KV_HEADS, TEST_HEAD_DIM),
            "allocate real QSA persistent state");
    const uint64_t demand_page = ds4_gpu_vmm_demand_page();
    if (demand_page != 0u) {
        REQUIRE(ds4_gpu_tensor_resident(
                    state.raw_index, 0,
                    (uint64_t)TEST_CONTEXT * TEST_INDEX_DIM * sizeof(float)) == 0u &&
                ds4_gpu_tensor_resident(
                    state.pooled_index, 0,
                    (uint64_t)(TEST_CONTEXT / TEST_RATIO) * TEST_INDEX_DIM *
                        sizeof(float)) == 0u &&
                ds4_gpu_tensor_resident(
                    state.k_cache, 0,
                    (uint64_t)TEST_CONTEXT * TEST_KV_DIM * sizeof(float)) == 0u &&
                ds4_gpu_tensor_resident(
                    state.v_cache, 0,
                    (uint64_t)TEST_CONTEXT * TEST_KV_DIM * sizeof(float)) == 0u,
                "QSA persistent state starts physically uncommitted");
    }
    ds4_gpu_tensor *dinput =
        ds4_gpu_tensor_alloc(input_count * sizeof(float));
    ds4_gpu_tensor *doutput =
        ds4_gpu_tensor_alloc(input_count * sizeof(float));
    REQUIRE(dinput && doutput, "real QSA device IO allocation");
    REQUIRE(ds4_gpu_tensor_write(
                dinput, 0, input, input_count * sizeof(float)),
            "real QSA input upload");
    REQUIRE(qwen4exp_qsa_forward(
                &ws, &state, &compact, &weights,
                dinput, doutput, TEST_ROWS, 0u),
            "complete real QSA forward");
    REQUIRE(state.length == TEST_ROWS, "real QSA state length");
    if (demand_page != 0u) {
        REQUIRE(ds4_gpu_tensor_resident(
                    state.raw_index, 0,
                    (uint64_t)TEST_CONTEXT * TEST_INDEX_DIM * sizeof(float)) != 0u &&
                ds4_gpu_tensor_resident(
                    state.pooled_index, 0,
                    (uint64_t)(TEST_CONTEXT / TEST_RATIO) * TEST_INDEX_DIM *
                        sizeof(float)) != 0u &&
                ds4_gpu_tensor_resident(
                    state.k_cache, 0,
                    (uint64_t)TEST_CONTEXT * TEST_KV_DIM * sizeof(float)) != 0u &&
                ds4_gpu_tensor_resident(
                    state.v_cache, 0,
                    (uint64_t)TEST_CONTEXT * TEST_KV_DIM * sizeof(float)) != 0u,
                "QSA forward commits every written cache range");
    }
    REQUIRE(ds4_gpu_tensor_read(ws.index_qk, 0, gpu_index_qk,
                                index_qk_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.index_query, 0, gpu_index_q,
                                index_q_count * sizeof(float)) &&
            ds4_gpu_tensor_read(state.raw_index, 0, gpu_raw,
                                (uint64_t)TEST_ROWS * TEST_INDEX_DIM *
                                sizeof(float)) &&
            ds4_gpu_tensor_read(state.pooled_index, 0, gpu_pool,
                                TEST_INDEX_DIM * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.q_projected, 0, gpu_qproj,
                                qproj_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.query, 0, gpu_query,
                                q_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.gate, 0, gpu_gate,
                                q_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.key, 0, gpu_key,
                                kv_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.value, 0, gpu_value,
                                kv_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.attention, 0, gpu_attention,
                                q_count * sizeof(float)) &&
            ds4_gpu_tensor_read(doutput, 0, gpu_output,
                                input_count * sizeof(float)) &&
            ds4_gpu_tensor_read(ws.selected_tokens, 0, selected,
                                sizeof(selected)) &&
            ds4_gpu_tensor_read(ws.selected_counts, 0, counts,
                                sizeof(counts)),
            "real QSA stage readback");

    qsa_compare("real Q8_0 index QK projection", gpu_index_qk,
                cpu_index_qk, index_qk_count, 1.5e-2, 2.0e-4, 0.08f);
    qsa_compare("real normalized/rotated index query", gpu_index_q,
                cpu_index_q, index_q_count, 2.0e-2, 3.0e-4, 0.08f);
    qsa_compare("real raw index-key cache", gpu_raw, cpu_raw,
                (uint64_t)TEST_ROWS * TEST_INDEX_DIM,
                1.5e-2, 2.0e-4, 0.08f);
    qsa_compare("real pooled four-token index key", gpu_pool, cpu_pool,
                TEST_INDEX_DIM, 2.0e-2, 3.0e-4, 0.08f);
    for (uint32_t row = 0; row < TEST_ROWS; row++) {
        REQUIRE(counts[row] == row + 1u,
                "real QSA causal selected count");
        for (uint32_t token = 0; token <= row; token++)
            REQUIRE(selected[(uint64_t)row * TEST_SELECTED_CAP + token] ==
                    (int32_t)token, "real QSA selected token order");
    }
    puts("real QSA causal token selection                    exact (1..5)");
    qsa_compare("real Q8_0 interleaved Q/gate projection", gpu_qproj,
                cpu_qproj, qproj_count, 1.5e-2, 2.0e-4, 0.08f);
    qsa_compare("real normalized/rotated query", gpu_query,
                cpu_query, q_count, 2.0e-2, 3.0e-4, 0.08f);
    qsa_compare("real sigmoid-gate projection slice", gpu_gate,
                cpu_gate, q_count, 1.5e-2, 2.0e-4, 0.08f);
    qsa_compare("real normalized/rotated key", gpu_key,
                cpu_key, kv_count, 2.0e-2, 3.0e-4, 0.08f);
    qsa_compare("real Q8_0 value projection", gpu_value,
                cpu_value, kv_count, 1.5e-2, 2.0e-4, 0.08f);
    qsa_compare("real selected GQA + sigmoid gate", gpu_attention,
                cpu_attention, q_count, 3.0e-2, 5.0e-4, 0.12f);
    qsa_compare("real complete QSA vs CPU dequant reference", gpu_output,
                cpu_output, input_count, 4.0e-2, 8.0e-4, 0.16f);

    const uint32_t chunks[] = {2u, 3u};
    REQUIRE(qwen4exp_qsa_state_reset(&state), "real QSA chunk state reset");
    qsa_run_chunks(&ws, &state, &compact, &weights,
                   dinput, doutput, chunks, 2u);
    REQUIRE(ds4_gpu_tensor_read(doutput, 0, chunk_output,
                                input_count * sizeof(float)),
            "real QSA chunk output readback");
    qsa_compare("real QSA arbitrary-chunk parity", chunk_output, gpu_output,
                input_count, 2.0e-3, 3.0e-6, 0.01f);
    const uint32_t decode[] = {1u, 1u, 1u, 1u, 1u};
    REQUIRE(qwen4exp_qsa_state_reset(&state), "real QSA decode state reset");
    qsa_run_chunks(&ws, &state, &compact, &weights,
                   dinput, doutput, decode, TEST_ROWS);
    REQUIRE(ds4_gpu_tensor_read(doutput, 0, chunk_output,
                                input_count * sizeof(float)),
            "real QSA decode output readback");
    qsa_compare("real QSA one-token decode parity", chunk_output, gpu_output,
                input_count, 2.0e-3, 3.0e-6, 0.01f);

    struct rusage usage;
    memset(&usage, 0, sizeof(usage));
    REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0, "read real QSA RSS");
    printf("real QSA: scratch=%.3f MiB state=%.3f MiB fixture=%.3f MiB "
           "peak-RSS=%ld KiB\n",
           (double)ws.bytes / 1048576.0,
           (double)state.bytes / 1048576.0,
           (double)fixture_bytes / 1048576.0, usage.ru_maxrss);

    REQUIRE(ds4_gpu_synchronize(), "final real QSA synchronization");
    ds4_gpu_tensor_free(doutput);
    ds4_gpu_tensor_free(dinput);
    qwen4exp_qsa_state_free(&state);
    qwen4exp_qsa_ws_free(&ws);
    ds4_gpu_unregister_model_map(fixture);
    ds4_gpu_cleanup();
    free(chunk_output); free(gpu_output); free(gpu_attention);
    free(gpu_value); free(gpu_key); free(gpu_gate); free(gpu_query);
    free(gpu_qproj); free(gpu_pool); free(gpu_raw); free(gpu_index_q);
    free(gpu_index_qk); free(cpu_output); free(cpu_attention);
    free(cpu_value); free(cpu_key); free(cpu_gate); free(cpu_query);
    free(cpu_qproj); free(cpu_pool); free(cpu_raw); free(cpu_index_q);
    free(cpu_index_qk); free(input); free(fixture);
    ds4_threads_shutdown();
    model_close(&model);
    puts("all real Qwen3.8 QSA checks passed");
    return 0;
}
