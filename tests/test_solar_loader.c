/* Solar Open2 GGUF architecture and metadata smoke.
 *
 * Usage: ./tests/test_solar_loader <first-model-shard.gguf> [--deepseek]
 *
 * This maps and validates metadata/tensor directories only.  It does not
 * allocate a session, touch model weights, or initialize a GPU.
 */
#include "../ds4.c"

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 ||
        (argc == 3 && strcmp(argv[2], "--deepseek") != 0)) {
        fprintf(stderr, "usage: %s <model.gguf> [--deepseek]\n", argv[0]);
        return 2;
    }

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
        printf("DeepSeek metadata: valid (%s)\n", DS4_MODEL_SHAPE_NAME);
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

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const bool want_gqa = (il % 4u) == 0u;
        if (ds4_solar_layer_is_gqa(il) != want_gqa) {
            fprintf(stderr, "Solar layer %u has the wrong attention branch\n", il);
            model_close(&model);
            return 1;
        }
    }

    printf("Solar metadata: valid (%u shards, 48 layers, 12 GQA, 36 KDA, "
           "NoPE)\n", model.split_count);
    model_close(&model);
    return 0;
}
