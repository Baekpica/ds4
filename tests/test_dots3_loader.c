/* dots3-note-prev metadata and full topology binder smoke.
 *
 *   ./tests/test_dots3_loader <dots3 first shard or merged gguf>
 */
#include "../ds4.c"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <dots3.gguf>\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DOTS3_NOTE ||
        DS4_MODEL_VARIANT != DS4_VARIANT_DOTS3_NOTE_PREV ||
        DS4_N_LAYER != 47 ||
        DS4_N_EMBD != 5120 ||
        DS4_N_VOCAB != 152064 ||
        DS4_N_HEAD != 128 ||
        DS4_N_KEY_MLA != 192 ||
        DS4_N_VALUE_MLA != 128 ||
        DS4_N_ROT != 64 ||
        DS4_N_LORA_Q != 1024 ||
        DS4_N_KV_LORA != 512 ||
        DS4_N_SWA_HEAD != 64 ||
        DS4_N_SWA_KV_LORA != 1024 ||
        DS4_N_SWA_KEY_MLA != 256 ||
        DS4_N_SWA != 513 ||
        DS4_N_EXPERT != 256 ||
        DS4_N_EXPERT_USED != 8 ||
        DS4_N_FF_EXP != 1536 ||
        DS4_N_FF_DENSE != 13824 ||
        DS4_N_LEADING_DENSE != 1 ||
        DS4_N_NEXTN_PREDICT != 1 ||
        DS4_N_INDEXER_HEAD != 64 ||
        DS4_N_INDEXER_HEAD_DIM != 128 ||
        DS4_N_INDEXER_TOP_K != 2048) {
        fprintf(stderr, "dots3-note profile selection produced unexpected constants\n");
        model_close(&model);
        return 1;
    }

    uint32_t full = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const bool expected = il + 1u < DS4_N_LAYER &&
                              (il == 0u || il % 4u == 1u);
        if (ds4_dots3_layer_is_full_attention(il) != expected) {
            fprintf(stderr, "dots3-note layer %u has the wrong attention branch\n", il);
            model_close(&model);
            return 1;
        }
        if (expected) full++;
    }
    if (full != 13) {
        fprintf(stderr, "dots3-note full-attention layer count is %u, expected 13\n", full);
        model_close(&model);
        return 1;
    }

    ds4_weights weights;
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    if (weights.layer[1].ffn_gate_exps->type != DS4_TENSOR_IQ2_XXS ||
        weights.layer[1].ffn_down_exps->type != DS4_TENSOR_Q2_K ||
        weights.layer[1].ffn_gate_inp->type != DS4_TENSOR_F32 ||
        weights.layer[1].attn_q_a_norm->type != DS4_TENSOR_Q8_0 ||
        weights.layer[1].attn_idx_k_norm->type != DS4_TENSOR_F32 ||
        weights.layer[1].attn_idx_k_norm_bias->type != DS4_TENSOR_F32 ||
        weights.layer[2].attn_idx_q_b != NULL ||
        !weights.layer[0].ffn_gate ||
        !weights.layer[46].nextn_eh_proj ||
        !weights.layer[46].nextn_shared_head_norm ||
        !weights.dots3_token_embd_mtp) {
        fprintf(stderr, "dots3-note quant policy or MTP binder is incomplete\n");
        model_close(&model);
        return 1;
    }

    /* The two MLA geometries must land in the bound tensor widths. */
    if (weights.layer[1].attn_q_b->dim[1] != 128u * 192u ||
        weights.layer[2].attn_q_b->dim[1] != 64u * 256u ||
        weights.layer[1].attn_kv_b->dim[0] != 512u ||
        weights.layer[2].attn_kv_b->dim[0] != 1024u ||
        weights.layer[1].attn_gate->dim[1] != 128u ||
        weights.layer[2].attn_gate->dim[1] != 64u ||
        weights.layer[46].attn_q_b->dim[1] != 64u * 256u) {
        fprintf(stderr, "dots3-note dual-geometry widths are wrong\n");
        model_close(&model);
        return 1;
    }

    printf("dots3-note-prev topology: valid "
           "(47 blocks, 13 full + 33 SWA + MTP, 256E top-8, DSA top-2048)\n");
    model_close(&model);
    return 0;
}
