/* Solar Open2 GGUF architecture and metadata smoke.
 *
 * Usage: ./tests/test_solar_loader <first-model-shard.gguf> [--deepseek]
 *
 * This maps and validates metadata/tensor directories only.  It does not
 * allocate a session, touch model weights, or initialize a GPU.
 */
#include "../ds4.c"

typedef struct {
    int sync_ok;
    int sync_calls;
    int trim_calls;
    char order[16];
    int order_len;
    uint64_t bank_bytes[4];
} solar_trim_fake;

static int solar_trim_fake_sync(void *user) {
    solar_trim_fake *f = user;
    f->sync_calls++;
    f->order[f->order_len++] = 'S';
    return f->sync_ok;
}

static uint64_t solar_trim_fake_bank(uint32_t bank, void *user) {
    solar_trim_fake *f = user;
    f->trim_calls++;
    f->order[f->order_len++] = (char)('0' + bank);
    return f->bank_bytes[bank];
}

static int test_solar_trim_sequence(void) {
    solar_trim_fake f = {.sync_ok = 0, .bank_bytes = {4u, 0u, 3u, 9u}};
    if (solar_trim_sequence_with_ops(
            4u, 6u, solar_trim_fake_sync, solar_trim_fake_bank, &f) != 0u ||
        f.sync_calls != 1 || f.trim_calls != 0 ||
        f.order_len != 1 || f.order[0] != 'S') {
        fprintf(stderr, "Solar trim ran an unmap after failed sync\n");
        return 0;
    }

    memset(&f, 0, sizeof(f));
    f.sync_ok = 1;
    f.bank_bytes[0] = 4u;
    f.bank_bytes[1] = 0u;
    f.bank_bytes[2] = 3u;
    f.bank_bytes[3] = 9u;
    const uint64_t freed = solar_trim_sequence_with_ops(
        4u, 6u, solar_trim_fake_sync, solar_trim_fake_bank, &f);
    if (freed != 7u || f.sync_calls != 1 || f.trim_calls != 3 ||
        f.order_len != 4 || memcmp(f.order, "S012", 4u) != 0) {
        fprintf(stderr, "Solar trim sync/unmap order regressed\n");
        return 0;
    }

    memset(&f, 0, sizeof(f));
    f.sync_ok = 1;
    if (solar_trim_sequence_with_ops(
            4u, 0u, solar_trim_fake_sync, solar_trim_fake_bank, &f) != 0u ||
        f.sync_calls != 0 || f.trim_calls != 0) {
        fprintf(stderr, "Solar zero-byte trim unexpectedly touched the device\n");
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 ||
        (argc == 3 && strcmp(argv[2], "--deepseek") != 0)) {
        fprintf(stderr, "usage: %s <model.gguf> [--deepseek]\n", argv[0]);
        return 2;
    }
    if (!test_solar_trim_sequence()) return 1;

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);

    if (argc == 3) {
        if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK4 ||
            (DS4_MODEL_VARIANT != DS4_VARIANT_FLASH &&
             DS4_MODEL_VARIANT != DS4_VARIANT_PRO) ||
            !DS4_USE_ROPE) {
            fprintf(stderr, "DeepSeek profile selection regressed\n");
            model_close(&model);
            return 1;
        }
        ds4_weights weights;
        weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
        printf("DeepSeek metadata and tensor layout: valid (%s)\n",
               DS4_MODEL_SHAPE_NAME);
        model_close(&model);
        return 0;
    }

    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_SOLAR_OPEN2 ||
        DS4_MODEL_VARIANT != DS4_VARIANT_SOLAR_OPEN2_250B ||
        DS4_N_LAYER != 48 ||
        DS4_N_EMBD != 4096 ||
        DS4_N_VOCAB != 196608 ||
        DS4_N_HEAD != 64 ||
        DS4_N_HEAD_KV != 8 ||
        DS4_N_HEAD_DIM != 128 ||
        DS4_N_KDA_HEAD_DIM != 128 ||
        DS4_N_SSM_CONV != 4 ||
        DS4_USE_ROPE ||
        fabsf(DS4_KDA_L2_EPS - 1.0e-6f) > 1.0e-12f ||
        fabsf(DS4_KDA_GATE_CLAMP_MIN - (-5.0f)) > 1.0e-6f) {
        fprintf(stderr, "Solar profile selection produced unexpected constants\n");
        model_close(&model);
        return 1;
    }

    ds4_weights weights;
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    if (weights.output_hc_base || weights.output_hc_fn ||
        weights.output_hc_scale) {
        fprintf(stderr, "Solar unexpectedly bound a hyper-connection output head\n");
        model_close(&model);
        return 1;
    }

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const bool want_gqa = (il % 4u) == 0u;
        if (ds4_solar_layer_is_gqa(il) != want_gqa) {
            fprintf(stderr, "Solar layer %u has the wrong attention branch\n", il);
            model_close(&model);
            return 1;
        }
        const ds4_layer_weights *layer = &weights.layer[il];
        if (!layer->attn_q || !layer->attn_k || !layer->attn_v ||
            !layer->attn_output ||
            (want_gqa && !layer->attn_gate) ||
            (!want_gqa && (!layer->ssm_q_conv || !layer->ssm_k_conv ||
                           !layer->ssm_v_conv || !layer->ssm_f_a ||
                           !layer->ssm_f_b || !layer->ssm_beta ||
                           !layer->ssm_a || !layer->ssm_dt_bias ||
                           !layer->ssm_g_a || !layer->ssm_g_b ||
                           !layer->ssm_o_norm))) {
            fprintf(stderr, "Solar layer %u did not bind its complete %s branch\n",
                    il, want_gqa ? "GQA" : "KDA");
            model_close(&model);
            return 1;
        }
    }

    /* The serving admission estimate must price the actual Solar runtime,
     * not DeepSeek's compressed-attention rings.  These constants are the
     * independent 262144-context calculation for hybrid K-FP8/V-FP4 KV and
     * a 2048-row prefill workspace, including the caller-owned KDA chunk
     * transform planes and pairwise matrix plus CUDA's 64-row grouped GQA
     * split-KV partials. */
    if (unsetenv("DS4_CUDA_SOLAR_GQA_GROUPED") != 0 ||
        unsetenv("DS4_CUDA_SOLAR_GQA_CHUNK") != 0 ||
        setenv("DS4_SOLAR_KV_FORMAT", "hybrid", 1) != 0 ||
        setenv("DS4_METAL_PREFILL_CHUNK", "2048", 1) != 0) {
        perror("Solar memory-plan environment");
        model_close(&model);
        return 1;
    }
    const ds4_context_memory memory =
        ds4_context_memory_estimate(DS4_BACKEND_CUDA, 262144);
    const uint64_t kda_chunk_scratch =
        ds4_gpu_solar_kda_prefill_scratch_bytes(2048u, 64u, 128u);
    if (memory.prefill_cap != 2048u ||
        memory.raw_cap != 262144u ||
        memory.raw_bytes != UINT64_C(4932501504) ||
        memory.compressed_bytes != UINT64_C(180513792) ||
        kda_chunk_scratch != UINT64_C(436207616) ||
        memory.scratch_bytes != UINT64_C(1901593664) ||
        memory.total_bytes != UINT64_C(7014608960)) {
        fprintf(stderr,
                "Solar 262144-context memory plan regressed: "
                "prefill=%u raw_cap=%u raw=%" PRIu64
                " state=%" PRIu64 " scratch=%" PRIu64
                " total=%" PRIu64 "\n",
                memory.prefill_cap, memory.raw_cap, memory.raw_bytes,
                memory.compressed_bytes, memory.scratch_bytes,
                memory.total_bytes);
        model_close(&model);
        return 1;
    }

    static const struct {
        const char *name;
        uint64_t raw_bytes;
        uint64_t total_bytes;
    } formats[] = {
        {"bf16", UINT64_C(12884901888), UINT64_C(14967009344)},
        {"fp8",  UINT64_C(6543114240),  UINT64_C(8625221696)},
        {"fp4",  UINT64_C(3321888768),  UINT64_C(5403996224)},
    };
    for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++) {
        if (setenv("DS4_SOLAR_KV_FORMAT", formats[i].name, 1) != 0) {
            perror("setenv");
            model_close(&model);
            return 1;
        }
        const ds4_context_memory variant =
            ds4_context_memory_estimate(DS4_BACKEND_CUDA, 262144);
        if (variant.raw_bytes != formats[i].raw_bytes ||
            variant.compressed_bytes != memory.compressed_bytes ||
            variant.scratch_bytes != memory.scratch_bytes ||
            variant.total_bytes != formats[i].total_bytes) {
            fprintf(stderr, "Solar %s KV memory plan regressed\n",
                    formats[i].name);
            model_close(&model);
            return 1;
        }
    }
    if (unsetenv("DS4_SOLAR_KV_FORMAT") != 0) {
        perror("unsetenv");
        model_close(&model);
        return 1;
    }
    const ds4_context_memory defaults =
        ds4_context_memory_estimate(DS4_BACKEND_CUDA, 262144);
    if (defaults.raw_bytes != memory.raw_bytes ||
        defaults.total_bytes != memory.total_bytes) {
        fprintf(stderr, "Solar default KV format is not hybrid\n");
        model_close(&model);
        return 1;
    }

    printf("Solar metadata and tensor layout: valid "
           "(%u shards, 48 layers, 12 GQA, 36 KDA, NoPE, "
           "262144 ctx %.2f GiB)\n",
           model.split_count,
           (double)memory.total_bytes / 1073741824.0);
    model_close(&model);
    return 0;
}
