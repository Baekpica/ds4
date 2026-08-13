/* Official-final Motif-3 metadata and full topology binder smoke.
 *
 *   ./tests/test_motif3_loader <motif3.gguf>
 */
#include "../ds4.c"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <motif3.gguf>\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MOTIF3 ||
        DS4_MODEL_VARIANT != DS4_VARIANT_MOTIF3 ||
        DS4_N_LAYER != 53 ||
        DS4_N_EMBD != 4096 ||
        DS4_N_VOCAB != 220160 ||
        DS4_N_HEAD != 80 ||
        DS4_N_HEAD_KV != 16 ||
        DS4_N_NOISE_HEAD != 16 ||
        DS4_N_HEAD_DIM != 192 ||
        DS4_N_VALUE_DIM != 128 ||
        DS4_N_EXPERT != 384 ||
        DS4_N_EXPERT_USED != 8 ||
        DS4_N_FF_EXP != 1280 ||
        DS4_N_LEADING_DENSE != 2 ||
        DS4_N_HC != 4 ||
        DS4_N_HC_SINKHORN_ITER != 20) {
        fprintf(stderr, "Motif-3 profile selection produced unexpected constants\n");
        model_close(&model);
        return 1;
    }

    uint32_t full = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const bool expected = il % 4u == 0u;
        if (ds4_motif3_layer_is_full_attention(il) != expected) {
            fprintf(stderr, "Motif-3 layer %u has the wrong attention branch\n", il);
            model_close(&model);
            return 1;
        }
        if (expected) full++;
    }
    if (full != 14) {
        fprintf(stderr, "Motif-3 full-attention layer count is %u, expected 14\n", full);
        model_close(&model);
        return 1;
    }

    ds4_weights weights;
    weights_bind(&weights, &model);
    if (weights.layer[2].ffn_gate_exps->type != DS4_TENSOR_IQ2_XXS ||
        weights.layer[2].ffn_down_exps->type != DS4_TENSOR_Q2_K ||
        weights.layer[2].ffn_gate_inp->type != DS4_TENSOR_F32 ||
        weights.layer[2].ffn_polynorm_exps_weight->type != DS4_TENSOR_F32 ||
        weights.layer[0].mhc_attn_proj_pre->type != DS4_TENSOR_BF16 ||
        !weights.motif_mtp_input_proj ||
        !weights.motif_mtp.ffn_polynorm_weight) {
        fprintf(stderr, "Motif-3 quant policy or MTP binder is incomplete\n");
        model_close(&model);
        return 1;
    }

    printf("Motif-3 official-final topology: valid "
           "(53 layers, 14 full + 39 SWA, 384E top-8, MTP present)\n");
    model_close(&model);
    return 0;
}
