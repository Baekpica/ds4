/* Solar Open 2 GGUF metadata and weight-layout smoke.
 *
 *   ./tests/test_solar_loader <model.gguf> [--weights]
 *
 * The default mode works with a vocab-only GGUF, so architecture recognition,
 * exact fixed dimensions, the per-layer S-L-L-L schedule, and the KDA semantic
 * constants can be tested before a 90 GiB artifact exists. --weights also
 * binds and validates every tensor in a completed mixed-quant artifact.
 */
#include "../ds4.c"

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 ||
        (argc == 3 && strcmp(argv[2], "--weights") != 0)) {
        fprintf(stderr, "usage: %s <solar-open2.gguf> [--weights]\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);

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

    if (argc == 3) {
        ds4_weights weights;
        weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
        puts("Solar full mixed-weight layout: valid");
    }

    printf("Solar metadata: valid (48 layers, 12 GQA, 36 KDA, NoPE, "
           "KDA eps=%.1e clamp=%.1f)\n",
           (double)DS4_KDA_L2_EPS, (double)DS4_KDA_GATE_CLAMP_MIN);
    model_close(&model);
    return 0;
}
