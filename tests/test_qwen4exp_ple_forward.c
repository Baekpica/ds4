/* Real-artifact H200 parity for the complete Qwen3.8 PLE injection path.
 *
 * The test copies only the six small resident PLE tensors into a compact,
 * page-aligned fixture.  The 90.96 GiB backbone therefore stays mmap-only,
 * while the embedding itself is gathered through the production bounded
 * SSD cache from the published 95.37 GiB BF16 sidecar.
 */
#include "../cuda/qwen38_ple.h"
#include "../ds4.c"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

enum {
    PLE_TEST_ROWS = 3,
    PLE_TEST_HIDDEN = 2560,
    PLE_TEST_HC = 4,
    PLE_TEST_WIDTH = PLE_TEST_HIDDEN * PLE_TEST_HC,
    PLE_TEST_KERNEL = 4,
    PLE_TEST_DILATION = 3,
    PLE_TEST_STATE = (PLE_TEST_KERNEL - 1) * PLE_TEST_DILATION,
};

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint64_t fixture_align(uint64_t value) {
    return (value + 4095u) & ~UINT64_C(4095);
}

static float bf16_bits_to_f32_test(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16u;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float sigmoid_test(float value) {
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
    printf("%-48s pass (worst abs %.3g, %.3fx limit)\n",
           name, worst_abs, worst_ratio);
}

static ds4_tensor fixture_copy_tensor(
        unsigned char *fixture, uint64_t fixture_bytes, uint64_t *cursor,
        const ds4_model *model, const ds4_tensor *source) {
    REQUIRE(source != NULL, "fixture source tensor exists");
    const uint64_t offset = fixture_align(*cursor);
    REQUIRE(offset <= fixture_bytes && source->bytes <= fixture_bytes - offset,
            "fixture tensor is in range");
    memcpy(fixture + offset, tensor_data(model, source), (size_t)source->bytes);
    ds4_tensor copy = *source;
    copy.rel_offset = offset;
    copy.abs_offset = offset;
    *cursor = offset + source->bytes;
    return copy;
}

static void cpu_group_norm(float *out, const float *input,
                           const float *weight, uint32_t width,
                           uint32_t group_size, uint32_t rows) {
    const uint32_t groups = width / group_size;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t group = 0; group < groups; group++) {
            const uint64_t base = (uint64_t)row * width +
                                  (uint64_t)group * group_size;
            double sum = 0.0;
            for (uint32_t d = 0; d < group_size; d++) {
                const double value = input[base + d];
                sum += value * value;
            }
            const float scale = 1.0f /
                sqrtf((float)(sum / (double)group_size) + DS4_RMS_EPS);
            const uint32_t weight_base = group * group_size;
            for (uint32_t d = 0; d < group_size; d++) {
                out[base + d] = input[base + d] * scale *
                                (1.0f + weight[weight_base + d]);
            }
        }
    }
}

static void cpu_ple_gate(float *gated, float *transformed,
                         const float *key_normed,
                         const float *query_normed,
                         const float *value) {
    for (uint32_t row = 0; row < PLE_TEST_ROWS; row++) {
        for (uint32_t hc = 0; hc < PLE_TEST_HC; hc++) {
            const uint64_t lane = (uint64_t)row * PLE_TEST_HC + hc;
            const uint64_t base = lane * PLE_TEST_HIDDEN;
            double dot = 0.0;
            for (uint32_t d = 0; d < PLE_TEST_HIDDEN; d++) {
                dot += (double)key_normed[base + d] *
                       query_normed[base + d];
            }
            const float raw =
                (float)(dot / sqrt((double)PLE_TEST_HIDDEN));
            const float sign = (float)((raw > 0.0f) - (raw < 0.0f));
            const float gate = sqrtf(fmaxf(fabsf(raw), 1.0e-6f)) * sign;
            transformed[lane] = gate;
            const float scale = sigmoid_test(gate);
            for (uint32_t d = 0; d < PLE_TEST_HIDDEN; d++) {
                gated[base + d] = scale *
                    value[(uint64_t)row * PLE_TEST_HIDDEN + d];
            }
        }
    }
}

static void cpu_ple_conv(float *out, float *state, const float *input,
                         const uint16_t *weight) {
    for (uint32_t token = 0; token < PLE_TEST_ROWS; token++) {
        for (uint32_t channel = 0; channel < PLE_TEST_WIDTH; channel++) {
            float sum = 0.0f;
            for (uint32_t tap = 0; tap < PLE_TEST_KERNEL; tap++) {
                const int32_t source_token =
                    (int32_t)token - PLE_TEST_STATE +
                    (int32_t)tap * PLE_TEST_DILATION;
                const float source = source_token >= 0
                    ? input[(uint64_t)source_token * PLE_TEST_WIDTH + channel]
                    : state[(uint64_t)channel * PLE_TEST_STATE +
                            (uint32_t)(PLE_TEST_STATE + source_token)];
                sum += source * bf16_bits_to_f32_test(
                    weight[(uint64_t)channel * PLE_TEST_KERNEL + tap]);
            }
            out[(uint64_t)token * PLE_TEST_WIDTH + channel] =
                sum * sigmoid_test(sum);
        }
    }
    for (uint32_t channel = 0; channel < PLE_TEST_WIDTH; channel++) {
        for (uint32_t slot = 0; slot < PLE_TEST_STATE; slot++) {
            const int32_t combined =
                PLE_TEST_ROWS + (int32_t)slot - PLE_TEST_STATE;
            state[(uint64_t)channel * PLE_TEST_STATE + slot] = combined >= 0
                ? input[(uint64_t)combined * PLE_TEST_WIDTH + channel]
                : 0.0f;
        }
    }
}

static void read_resident_embedding(const ds4_model *resident_model,
                                    const uint64_t *row_ids,
                                    uint16_t *embedding,
                                    uint64_t expected_padded_rows) {
    ds4_tensor *parts[DS4_PLE_N_LOGICAL_PARTS];
    uint64_t starts[DS4_PLE_N_LOGICAL_PARTS + 1u];
    starts[0] = 0;
    for (uint32_t part = 0; part < DS4_PLE_N_LOGICAL_PARTS; part++) {
        char name[96];
        const int written = snprintf(
            name, sizeof(name), "blk.1.ple.ngram_embd.part_%03u.weight", part);
        REQUIRE(written > 0 && (size_t)written < sizeof(name),
                "resident PLE tensor name fits");
        parts[part] = model_find_tensor(resident_model, name);
        REQUIRE(parts[part] != NULL, "resident BF16 PLE tensor exists");
        REQUIRE(parts[part]->type == DS4_TENSOR_BF16 &&
                    parts[part]->ndim == 2u &&
                    parts[part]->dim[0] == DS4_PLE_ROW_DIM,
                "resident PLE tensor has BF16 row-major layout");
        REQUIRE(parts[part]->dim[1] <= UINT64_MAX / DS4_PLE_ROW_BYTES &&
                    parts[part]->bytes ==
                        parts[part]->dim[1] * DS4_PLE_ROW_BYTES,
                "resident PLE tensor byte count matches its rows");
        REQUIRE(starts[part] <= UINT64_MAX - parts[part]->dim[1],
                "resident PLE row count does not overflow");
        starts[part + 1u] = starts[part] + parts[part]->dim[1];
    }
    REQUIRE(starts[DS4_PLE_N_LOGICAL_PARTS] == expected_padded_rows,
            "resident BF16 GGUF and SSD manifest expose the same PLE rows");

    for (uint32_t index = 0;
         index < PLE_TEST_ROWS * DS4_PLE_N_HEADS; index++) {
        const uint64_t row = row_ids[index];
        REQUIRE(row < expected_padded_rows,
                "resident PLE row ID is in range");
        uint32_t low = 0;
        uint32_t high = DS4_PLE_N_LOGICAL_PARTS;
        while (low + 1u < high) {
            const uint32_t mid = low + (high - low) / 2u;
            if (starts[mid] <= row)
                low = mid;
            else
                high = mid;
        }
        const uint64_t local = row - starts[low];
        REQUIRE(local < parts[low]->dim[1],
                "resident PLE logical-part lookup is in range");
        memcpy((unsigned char *)embedding +
                   (uint64_t)index * DS4_PLE_ROW_BYTES,
               (const unsigned char *)tensor_data(resident_model, parts[low]) +
                   local * DS4_PLE_ROW_BYTES,
               DS4_PLE_ROW_BYTES);
    }
}

static void require_bit_exact(const char *name, const void *got,
                              const void *want, uint64_t bytes) {
    if (memcmp(got, want, (size_t)bytes) != 0) {
        const unsigned char *g = (const unsigned char *)got;
        const unsigned char *w = (const unsigned char *)want;
        uint64_t at = 0;
        while (at < bytes && g[at] == w[at]) at++;
        fprintf(stderr,
                "FAIL: %s differs at byte %llu: got=%02x want=%02x\n",
                name, (unsigned long long)at,
                at < bytes ? g[at] : 0u, at < bytes ? w[at] : 0u);
        exit(1);
    }
    printf("%-48s pass (%llu bytes exact)\n", name,
           (unsigned long long)bytes);
}

static void run_gpu_ple_forward(
        const unsigned char *fixture, uint64_t fixture_bytes,
        const ds4_tensor *key, const ds4_tensor *value,
        const ds4_tensor *key_norm, const ds4_tensor *query_norm,
        const ds4_tensor *conv_norm, const ds4_tensor *conv,
        ds4_gpu_tensor *dembed_bf16, ds4_gpu_tensor *dembed,
        ds4_gpu_tensor *dkey, ds4_gpu_tensor *dvalue,
        ds4_gpu_tensor *dquery, ds4_gpu_tensor *dkey_normed,
        ds4_gpu_tensor *dquery_normed, ds4_gpu_tensor *dgated,
        ds4_gpu_tensor *dgate, ds4_gpu_tensor *dconv_normed,
        ds4_gpu_tensor *dconv, ds4_gpu_tensor *doutput,
        ds4_gpu_tensor *dstate) {
    const uint64_t embed_values =
        (uint64_t)PLE_TEST_ROWS * PLE_TEST_HIDDEN;
    const uint64_t hc_values =
        (uint64_t)PLE_TEST_ROWS * PLE_TEST_WIDTH;
    const uint64_t state_values =
        (uint64_t)PLE_TEST_WIDTH * PLE_TEST_STATE;
    REQUIRE(ds4_gpu_tensor_fill_f32(dstate, 0.0f, state_values),
            "clear PLE convolution state");
    REQUIRE(ds4_gpu_qwen4exp_bf16_to_f32_tensor(
                dembed, dembed_bf16, embed_values),
            "promote PLE activation");
    REQUIRE(ds4_gpu_matmul_q8_0_pair_tensor(
                dkey, dvalue, fixture, fixture_bytes,
                key->abs_offset, value->abs_offset,
                PLE_TEST_HIDDEN, PLE_TEST_WIDTH, PLE_TEST_HIDDEN,
                dembed, PLE_TEST_ROWS),
            "paired Q8_0 PLE key/value projection");
    REQUIRE(ds4_gpu_qwen4exp_group_rms_norm_rows_tensor(
                dkey_normed, dkey, fixture, fixture_bytes,
                key_norm->abs_offset, PLE_TEST_WIDTH, PLE_TEST_HIDDEN,
                PLE_TEST_ROWS, DS4_RMS_EPS),
            "PLE key grouped RMSNorm");
    REQUIRE(ds4_gpu_qwen4exp_group_rms_norm_rows_tensor(
                dquery_normed, dquery, fixture, fixture_bytes,
                query_norm->abs_offset, PLE_TEST_WIDTH, PLE_TEST_HIDDEN,
                PLE_TEST_ROWS, DS4_RMS_EPS),
            "PLE query grouped RMSNorm");
    REQUIRE(ds4_gpu_qwen4exp_ple_gate_tensor(
                dgated, dgate, dkey_normed, dquery_normed, dvalue,
                PLE_TEST_ROWS, PLE_TEST_HIDDEN, PLE_TEST_HC),
            "PLE signed-sqrt gate");
    REQUIRE(ds4_gpu_qwen4exp_group_rms_norm_rows_tensor(
                dconv_normed, dgated, fixture, fixture_bytes,
                conv_norm->abs_offset, PLE_TEST_WIDTH, PLE_TEST_HIDDEN,
                PLE_TEST_ROWS, DS4_RMS_EPS),
            "PLE convolution input grouped RMSNorm");
    REQUIRE(ds4_gpu_qwen4exp_ple_conv_tensor(
                dconv, dstate, dconv_normed, fixture, fixture_bytes,
                conv->abs_offset, PLE_TEST_ROWS, PLE_TEST_WIDTH,
                PLE_TEST_KERNEL, PLE_TEST_DILATION),
            "PLE dilated causal convolution");
    REQUIRE(ds4_gpu_add_tensor(doutput, dgated, dconv,
                               (uint32_t)hc_values),
            "PLE gated residual plus convolution");
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s SSD_FIRST_GGUF ARTIFACT_ROOT "
                "RESIDENT_BF16_FIRST_GGUF\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN4EXP,
            "Qwen4Exp model profile");
    ds4_weights weights;
    weights_bind(&weights, &model, false, 0, DS4_N_LAYER - 1u, true, false);
    const ds4_qwen_ple_weights *ple = &weights.layer[1].qwen_ple;
    REQUIRE(ple->key->type == DS4_TENSOR_Q8_0 &&
            ple->value->type == DS4_TENSOR_Q8_0,
            "published PLE projections are Q8_0");
    REQUIRE(ple->key_norm->type == DS4_TENSOR_F32 &&
            ple->query_norm->type == DS4_TENSOR_F32 &&
            ple->conv_norm->type == DS4_TENSOR_F32 &&
            ple->conv->type == DS4_TENSOR_BF16,
            "published PLE numerical tensor policy");

    const ds4_tensor *sources[] = {
        ple->key, ple->value, ple->key_norm,
        ple->query_norm, ple->conv_norm, ple->conv,
    };
    uint64_t fixture_bytes = 4096u;
    for (size_t i = 0; i < sizeof(sources) / sizeof(sources[0]); i++) {
        fixture_bytes = fixture_align(fixture_bytes) + sources[i]->bytes;
    }
    fixture_bytes = fixture_align(fixture_bytes);
    unsigned char *fixture = NULL;
    REQUIRE(posix_memalign((void **)&fixture, 4096u,
                           (size_t)fixture_bytes) == 0,
            "compact PLE weight fixture allocation");
    memset(fixture, 0, (size_t)fixture_bytes);
    uint64_t fixture_cursor = 4096u;
    ds4_tensor key = fixture_copy_tensor(
        fixture, fixture_bytes, &fixture_cursor, &model, ple->key);
    ds4_tensor value = fixture_copy_tensor(
        fixture, fixture_bytes, &fixture_cursor, &model, ple->value);
    ds4_tensor key_norm = fixture_copy_tensor(
        fixture, fixture_bytes, &fixture_cursor, &model, ple->key_norm);
    ds4_tensor query_norm = fixture_copy_tensor(
        fixture, fixture_bytes, &fixture_cursor, &model, ple->query_norm);
    ds4_tensor conv_norm = fixture_copy_tensor(
        fixture, fixture_bytes, &fixture_cursor, &model, ple->conv_norm);
    ds4_tensor conv = fixture_copy_tensor(
        fixture, fixture_bytes, &fixture_cursor, &model, ple->conv);
    printf("compact resident PLE fixture: %.3f MiB (backbone mmap %.3f GiB)\n",
           (double)fixture_bytes / (1024.0 * 1024.0),
           (double)model.size / (1024.0 * 1024.0 * 1024.0));

    ds4_model fixture_model;
    memset(&fixture_model, 0, sizeof(fixture_model));
    fixture_model.fd = -1;
    fixture_model.map = fixture;
    fixture_model.size = fixture_bytes;

    char error[512] = {0};
    ds4_ple_store *store = ds4_ple_store_open(
        argv[2], "ple/ple-manifest.json", 16u * 1024u, 4u, true,
        error, sizeof(error));
    REQUIRE(store != NULL, error[0] ? error : "open BF16 SSD-PLE store");
    ds4_qwen38_ple_cuda *ple_cuda =
        ds4_qwen38_ple_cuda_create(store, error, sizeof(error));
    REQUIRE(ple_cuda != NULL,
            error[0] ? error : "create bounded PLE CUDA gather");

    const ds4_ple_hash_config *hash = ds4_ple_store_hash_config(store);
    int64_t tokens[PLE_TEST_ROWS] = {
        104857u, hash->eos_token_id, 4242u,
    };
    uint64_t row_ids[PLE_TEST_ROWS * DS4_PLE_N_HEADS];
    ds4_ple_hash_state hash_state;
    ds4_ple_hash_state_reset(&hash_state, hash);
    REQUIRE(ds4_ple_hash_rows(hash, &hash_state, tokens, PLE_TEST_ROWS,
                              row_ids, error, sizeof(error)),
            error[0] ? error : "derive exact PLE row IDs");

    ds4_model resident_model;
    model_open(&resident_model, argv[3], false, false);
    config_validate_model(&resident_model);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN4EXP,
            "resident reference uses the same Qwen4Exp family");

    const uint64_t embed_values =
        (uint64_t)PLE_TEST_ROWS * PLE_TEST_HIDDEN;
    const uint64_t hc_values =
        (uint64_t)PLE_TEST_ROWS * PLE_TEST_WIDTH;
    const uint64_t value_values = embed_values;
    const uint64_t lane_values =
        (uint64_t)PLE_TEST_ROWS * PLE_TEST_HC;
    const uint64_t state_values =
        (uint64_t)PLE_TEST_WIDTH * PLE_TEST_STATE;

    uint16_t *embedding_bf16 =
        (uint16_t *)malloc(embed_values * sizeof(*embedding_bf16));
    uint16_t *embedding_gpu_bf16 =
        (uint16_t *)malloc(embed_values * sizeof(*embedding_gpu_bf16));
    float *embedding = (float *)malloc(embed_values * sizeof(*embedding));
    float *key_cpu = (float *)malloc(hc_values * sizeof(*key_cpu));
    float *key_gpu = (float *)malloc(hc_values * sizeof(*key_gpu));
    float *value_cpu = (float *)malloc(value_values * sizeof(*value_cpu));
    float *value_gpu = (float *)malloc(value_values * sizeof(*value_gpu));
    float *query = (float *)malloc(hc_values * sizeof(*query));
    float *key_normed_cpu =
        (float *)malloc(hc_values * sizeof(*key_normed_cpu));
    float *query_normed_cpu =
        (float *)malloc(hc_values * sizeof(*query_normed_cpu));
    float *gated_cpu = (float *)malloc(hc_values * sizeof(*gated_cpu));
    float *gated_gpu = (float *)malloc(hc_values * sizeof(*gated_gpu));
    float *gate_cpu = (float *)malloc(lane_values * sizeof(*gate_cpu));
    float *gate_gpu = (float *)malloc(lane_values * sizeof(*gate_gpu));
    float *conv_normed_cpu =
        (float *)malloc(hc_values * sizeof(*conv_normed_cpu));
    float *conv_normed_gpu =
        (float *)malloc(hc_values * sizeof(*conv_normed_gpu));
    float *conv_cpu = (float *)malloc(hc_values * sizeof(*conv_cpu));
    float *conv_gpu = (float *)malloc(hc_values * sizeof(*conv_gpu));
    float *output_cpu = (float *)malloc(hc_values * sizeof(*output_cpu));
    float *output_gpu = (float *)malloc(hc_values * sizeof(*output_gpu));
    float *state_cpu = (float *)calloc(state_values, sizeof(*state_cpu));
    float *state_gpu = (float *)malloc(state_values * sizeof(*state_gpu));
    float *key_resident = (float *)malloc(hc_values * sizeof(*key_resident));
    float *value_resident =
        (float *)malloc(value_values * sizeof(*value_resident));
    float *gate_resident =
        (float *)malloc(lane_values * sizeof(*gate_resident));
    float *gated_resident =
        (float *)malloc(hc_values * sizeof(*gated_resident));
    float *conv_normed_resident =
        (float *)malloc(hc_values * sizeof(*conv_normed_resident));
    float *conv_resident =
        (float *)malloc(hc_values * sizeof(*conv_resident));
    float *output_resident =
        (float *)malloc(hc_values * sizeof(*output_resident));
    float *state_resident =
        (float *)malloc(state_values * sizeof(*state_resident));
    REQUIRE(embedding_bf16 && embedding_gpu_bf16 && embedding && key_cpu &&
            key_gpu && value_cpu && value_gpu && query && key_normed_cpu &&
            query_normed_cpu && gated_cpu && gated_gpu && gate_cpu &&
            gate_gpu && conv_normed_cpu && conv_normed_gpu && conv_cpu && conv_gpu &&
            output_cpu && output_gpu && state_cpu && state_gpu &&
            key_resident && value_resident && gate_resident &&
            gated_resident && conv_normed_resident && conv_resident &&
            output_resident && state_resident,
            "PLE forward host allocations");

    read_resident_embedding(
        &resident_model, row_ids, embedding_bf16,
        ds4_ple_store_layout(store)->padded_vocabulary_rows);
    model_release_mapping_cache(&resident_model);
    model_close(&resident_model);
    for (uint64_t i = 0; i < embed_values; i++)
        embedding[i] = bf16_bits_to_f32_test(embedding_bf16[i]);
    for (uint64_t i = 0; i < hc_values; i++) {
        query[i] = 0.41f * sinf((float)(i + 5u) * 0.0037f) +
                   0.17f * cosf((float)(i + 29u) * 0.011f);
    }

    matmul_q8_0_batch(key_cpu, &fixture_model, &key,
                      embedding, PLE_TEST_ROWS);
    matmul_q8_0_batch(value_cpu, &fixture_model, &value,
                      embedding, PLE_TEST_ROWS);
    cpu_group_norm(key_normed_cpu, key_cpu,
                   (const float *)tensor_data(&fixture_model, &key_norm),
                   PLE_TEST_WIDTH, PLE_TEST_HIDDEN, PLE_TEST_ROWS);
    cpu_group_norm(query_normed_cpu, query,
                   (const float *)tensor_data(&fixture_model, &query_norm),
                   PLE_TEST_WIDTH, PLE_TEST_HIDDEN, PLE_TEST_ROWS);
    cpu_ple_gate(gated_cpu, gate_cpu, key_normed_cpu,
                 query_normed_cpu, value_cpu);
    cpu_group_norm(conv_normed_cpu, gated_cpu,
                   (const float *)tensor_data(&fixture_model, &conv_norm),
                   PLE_TEST_WIDTH, PLE_TEST_HIDDEN, PLE_TEST_ROWS);
    cpu_ple_conv(conv_cpu, state_cpu, conv_normed_cpu,
                 (const uint16_t *)tensor_data(&fixture_model, &conv));
    for (uint64_t i = 0; i < hc_values; i++)
        output_cpu[i] = gated_cpu[i] + conv_cpu[i];

    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(unsetenv("DS4_CUDA_COPY_MODEL") == 0,
            "disable whole-fixture copy override");
    REQUIRE(ds4_gpu_set_model_map(fixture, fixture_bytes),
            "register compact PLE weight fixture");

    ds4_gpu_tensor *dembed_bf16 =
        ds4_gpu_tensor_alloc(embed_values * sizeof(uint16_t));
    ds4_gpu_tensor *dembed = ds4_gpu_tensor_alloc(embed_values * sizeof(float));
    ds4_gpu_tensor *dkey = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dvalue = ds4_gpu_tensor_alloc(value_values * sizeof(float));
    ds4_gpu_tensor *dquery = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dkey_normed =
        ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dquery_normed =
        ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dgated = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dgate = ds4_gpu_tensor_alloc(lane_values * sizeof(float));
    ds4_gpu_tensor *dconv_normed =
        ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dconv = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *doutput = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dstate = ds4_gpu_tensor_alloc(state_values * sizeof(float));
    REQUIRE(dembed_bf16 && dembed && dkey && dvalue && dquery &&
            dkey_normed && dquery_normed && dgated && dgate &&
            dconv_normed && dconv && doutput && dstate,
            "PLE forward GPU allocations");
    REQUIRE(ds4_gpu_tensor_write(dquery, 0, query,
                                 hc_values * sizeof(float)),
            "upload PLE query streams");
    REQUIRE(ds4_gpu_tensor_write(dembed_bf16, 0, embedding_bf16,
                                 embed_values * sizeof(uint16_t)),
            "upload resident BF16 PLE activation");
    run_gpu_ple_forward(
        fixture, fixture_bytes, &key, &value, &key_norm, &query_norm,
        &conv_norm, &conv, dembed_bf16, dembed, dkey, dvalue, dquery,
        dkey_normed, dquery_normed, dgated, dgate, dconv_normed, dconv,
        doutput, dstate);
    REQUIRE(ds4_gpu_tensor_read(dkey, 0, key_resident,
                                hc_values * sizeof(float)),
            "download resident PLE key projection");
    REQUIRE(ds4_gpu_tensor_read(dvalue, 0, value_resident,
                                value_values * sizeof(float)),
            "download resident PLE value projection");
    REQUIRE(ds4_gpu_tensor_read(dgate, 0, gate_resident,
                                lane_values * sizeof(float)),
            "download resident PLE gate");
    REQUIRE(ds4_gpu_tensor_read(dgated, 0, gated_resident,
                                hc_values * sizeof(float)),
            "download resident PLE gated value");
    REQUIRE(ds4_gpu_tensor_read(dconv_normed, 0, conv_normed_resident,
                                hc_values * sizeof(float)),
            "download resident PLE convolution input");
    REQUIRE(ds4_gpu_tensor_read(dconv, 0, conv_resident,
                                hc_values * sizeof(float)),
            "download resident PLE convolution");
    REQUIRE(ds4_gpu_tensor_read(doutput, 0, output_resident,
                                hc_values * sizeof(float)),
            "download resident PLE output");
    REQUIRE(ds4_gpu_tensor_read(dstate, 0, state_resident,
                                state_values * sizeof(float)),
            "download resident PLE convolution state");

    /* First cold gather validates source delivery. Then force far more unique
     * pages through the deliberately tiny four-page cache and require the
     * original rows to be physically re-read before the offloaded forward. */
    REQUIRE(ds4_qwen38_ple_cuda_gather(
                ple_cuda, row_ids, PLE_TEST_ROWS,
                (void *)ds4_gpu_tensor_ptr(dembed_bf16), NULL,
                error, sizeof(error)),
            error[0] ? error : "first bounded SSD PLE CUDA gather");
    REQUIRE(ds4_gpu_tensor_read(dembed_bf16, 0, embedding_gpu_bf16,
                                embed_values * sizeof(uint16_t)),
            "download first gathered BF16 embedding");
    require_bit_exact("resident GGUF -> first SSD gather",
                      embedding_gpu_bf16, embedding_bf16,
                      embed_values * sizeof(uint16_t));

    unsigned char churn_row[DS4_PLE_ROW_BYTES];
    const uint64_t usable_rows =
        ds4_ple_store_layout(store)->usable_vocabulary_rows;
    uint64_t generator = UINT64_C(0xd1b54a32d192ed03);
    for (uint32_t i = 0; i < 257u; i++) {
        generator = generator * UINT64_C(6364136223846793005) + 1u;
        const uint64_t row = generator % usable_rows;
        REQUIRE(ds4_ple_store_read_row(
                    store, row, churn_row, sizeof(churn_row),
                    error, sizeof(error)),
                error[0] ? error : "churn bounded SSD PLE cache");
    }
    ds4_ple_stats stats_before_reload;
    ds4_ple_store_get_stats(store, &stats_before_reload);
    REQUIRE(stats_before_reload.cache_evictions > 0u,
            "tiny SSD PLE cache exercised eviction");

    REQUIRE(ds4_qwen38_ple_cuda_gather(
                ple_cuda, row_ids, PLE_TEST_ROWS,
                (void *)ds4_gpu_tensor_ptr(dembed_bf16), NULL,
                error, sizeof(error)),
            error[0] ? error : "reloaded bounded SSD PLE CUDA gather");
    REQUIRE(ds4_gpu_tensor_read(dembed_bf16, 0, embedding_gpu_bf16,
                                embed_values * sizeof(uint16_t)),
            "download reloaded BF16 embedding");
    require_bit_exact("resident GGUF -> SSD gather after eviction",
                      embedding_gpu_bf16, embedding_bf16,
                      embed_values * sizeof(uint16_t));
    ds4_ple_stats stats_after_reload;
    ds4_ple_store_get_stats(store, &stats_after_reload);
    REQUIRE(stats_after_reload.read_operations >
                stats_before_reload.read_operations,
            "post-eviction PLE gather physically reloaded pages");

    run_gpu_ple_forward(
        fixture, fixture_bytes, &key, &value, &key_norm, &query_norm,
        &conv_norm, &conv, dembed_bf16, dembed, dkey, dvalue, dquery,
        dkey_normed, dquery_normed, dgated, dgate, dconv_normed, dconv,
        doutput, dstate);

    REQUIRE(ds4_gpu_tensor_read(dkey, 0, key_gpu,
                                hc_values * sizeof(float)),
            "download PLE key projection");
    REQUIRE(ds4_gpu_tensor_read(dvalue, 0, value_gpu,
                                value_values * sizeof(float)),
            "download PLE value projection");
    REQUIRE(ds4_gpu_tensor_read(dgate, 0, gate_gpu,
                                lane_values * sizeof(float)),
            "download PLE gate");
    REQUIRE(ds4_gpu_tensor_read(dgated, 0, gated_gpu,
                                hc_values * sizeof(float)),
            "download PLE gated value");
    REQUIRE(ds4_gpu_tensor_read(dconv, 0, conv_gpu,
                                hc_values * sizeof(float)),
            "download PLE convolution");
    REQUIRE(ds4_gpu_tensor_read(dconv_normed, 0, conv_normed_gpu,
                                hc_values * sizeof(float)),
            "download normalized PLE convolution input");
    REQUIRE(ds4_gpu_tensor_read(doutput, 0, output_gpu,
                                hc_values * sizeof(float)),
            "download PLE output");
    REQUIRE(ds4_gpu_tensor_read(dstate, 0, state_gpu,
                                state_values * sizeof(float)),
            "download PLE convolution state");

    require_bit_exact("resident vs SSD key projection", key_gpu,
                      key_resident, hc_values * sizeof(float));
    require_bit_exact("resident vs SSD value projection", value_gpu,
                      value_resident, value_values * sizeof(float));
    require_bit_exact("resident vs SSD transformed gate", gate_gpu,
                      gate_resident, lane_values * sizeof(float));
    require_bit_exact("resident vs SSD gated value", gated_gpu,
                      gated_resident, hc_values * sizeof(float));
    require_bit_exact("resident vs SSD convolution input", conv_normed_gpu,
                      conv_normed_resident, hc_values * sizeof(float));
    require_bit_exact("resident vs SSD convolution", conv_gpu,
                      conv_resident, hc_values * sizeof(float));
    require_bit_exact("resident vs SSD final PLE injection", output_gpu,
                      output_resident, hc_values * sizeof(float));
    require_bit_exact("resident vs SSD PLE persistent state", state_gpu,
                      state_resident, state_values * sizeof(float));

    compare_f32("real Q8_0 PLE key projection", key_gpu, key_cpu,
                hc_values, 2.0e-4f, 2.0e-4f);
    compare_f32("real Q8_0 PLE value projection", value_gpu, value_cpu,
                value_values, 2.0e-4f, 2.0e-4f);
    compare_f32("real PLE signed-sqrt gate", gate_gpu, gate_cpu,
                lane_values, 4.0e-4f, 4.0e-4f);
    compare_f32("real PLE gated value", gated_gpu, gated_cpu,
                hc_values, 4.0e-4f, 4.0e-4f);
    compare_f32("real PLE convolution input norm", conv_normed_gpu,
                conv_normed_cpu, hc_values, 2.0e-3f, 2.0e-3f);
    compare_f32("real PLE BF16 dilated convolution", conv_gpu, conv_cpu,
                hc_values, 6.0e-4f, 6.0e-4f);
    compare_f32("real end-to-end SSD-PLE injection", output_gpu, output_cpu,
                hc_values, 8.0e-4f, 8.0e-4f);
    /* The state update is a copy, not arithmetic. Compare it against the
     * exact GPU convolution input so projection/norm roundoff does not turn a
     * state-lifetime check into a second numerical-parity check. */
    memset(state_cpu, 0, state_values * sizeof(*state_cpu));
    for (uint32_t channel = 0; channel < PLE_TEST_WIDTH; channel++) {
        for (uint32_t row = 0; row < PLE_TEST_ROWS; row++) {
            state_cpu[(uint64_t)channel * PLE_TEST_STATE +
                      (PLE_TEST_STATE - PLE_TEST_ROWS + row)] =
                conv_normed_gpu[(uint64_t)row * PLE_TEST_WIDTH + channel];
        }
    }
    compare_f32("real PLE final convolution state", state_gpu, state_cpu,
                state_values, 0.0f, 0.0f);

    ds4_qwen38_ple_cuda_stats cuda_stats;
    ds4_ple_stats store_stats;
    ds4_qwen38_ple_cuda_get_stats(ple_cuda, &cuda_stats);
    ds4_ple_store_get_stats(store, &store_stats);
    const ds4_ple_layout *layout = ds4_ple_store_layout(store);
    struct rusage usage;
    memset(&usage, 0, sizeof(usage));
    REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0, "read process RSS");
    printf("SSD-PLE: cache=%zu rows=%llu output=%llu bytes direct=%u/%u "
           "reads=%llu physical=%llu evictions=%llu reload-reads=%llu "
           "peak-RSS=%ld KiB\n",
           layout->cache_bytes,
           (unsigned long long)cuda_stats.gathered_rows,
           (unsigned long long)cuda_stats.output_bytes,
           layout->direct_io_file_count, layout->physical_file_count,
           (unsigned long long)store_stats.read_operations,
           (unsigned long long)store_stats.physical_bytes,
           (unsigned long long)store_stats.cache_evictions,
           (unsigned long long)(stats_after_reload.read_operations -
                                stats_before_reload.read_operations),
           usage.ru_maxrss);

    REQUIRE(ds4_gpu_synchronize(), "final PLE forward synchronization");
    ds4_gpu_tensor_free(dstate);
    ds4_gpu_tensor_free(doutput);
    ds4_gpu_tensor_free(dconv);
    ds4_gpu_tensor_free(dconv_normed);
    ds4_gpu_tensor_free(dgate);
    ds4_gpu_tensor_free(dgated);
    ds4_gpu_tensor_free(dquery_normed);
    ds4_gpu_tensor_free(dkey_normed);
    ds4_gpu_tensor_free(dquery);
    ds4_gpu_tensor_free(dvalue);
    ds4_gpu_tensor_free(dkey);
    ds4_gpu_tensor_free(dembed);
    ds4_gpu_tensor_free(dembed_bf16);
    ds4_qwen38_ple_cuda_destroy(ple_cuda);
    ds4_ple_store_close(store);
    ds4_gpu_unregister_model_map(fixture);
    ds4_gpu_cleanup();
    ds4_threads_shutdown();
    model_close(&model);

    free(state_resident);
    free(output_resident);
    free(conv_resident);
    free(conv_normed_resident);
    free(gated_resident);
    free(gate_resident);
    free(value_resident);
    free(key_resident);
    free(state_gpu);
    free(state_cpu);
    free(output_gpu);
    free(output_cpu);
    free(conv_gpu);
    free(conv_cpu);
    free(conv_normed_gpu);
    free(conv_normed_cpu);
    free(gate_gpu);
    free(gate_cpu);
    free(gated_gpu);
    free(gated_cpu);
    free(query_normed_cpu);
    free(key_normed_cpu);
    free(query);
    free(value_gpu);
    free(value_cpu);
    free(key_gpu);
    free(key_cpu);
    free(embedding);
    free(embedding_gpu_bf16);
    free(embedding_bf16);
    free(fixture);
    puts("all real-artifact SSD-PLE forward checks passed");
    return 0;
}
