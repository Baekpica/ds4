/* Qwen3.8-Flash-Next pinned metadata, hybrid schedule, and SSD-PLE contract
 * smoke.
 *
 *   ./tests/test_qwen4exp_loader <first BF16/Q8/Mixed GGUF shard>
 */
#include "../ds4.c"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <qwen4exp.gguf>\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_QWEN4EXP ||
        DS4_MODEL_VARIANT != DS4_VARIANT_QWEN38_FLASH_NEXT ||
        DS4_N_LAYER != 48 || DS4_N_EMBD != 2560 ||
        DS4_N_VOCAB != 248320 || DS4_N_HEAD != 24 ||
        DS4_N_HEAD_KV != 2 || DS4_N_HEAD_DIM != 256 ||
        DS4_N_ROT != 64 || DS4_N_EXPERT != 512 ||
        DS4_N_EXPERT_USED != 10 || DS4_N_FF_EXP != 640 ||
        DS4_N_FF_SHEXP != 640 || DS4_N_HC != 4 ||
        DS4_N_INDEXER_HEAD != 4 || DS4_N_INDEXER_HEAD_DIM != 128 ||
        DS4_N_INDEXER_TOP_K != 2048 || DS4_N_FULL_ATTN_COUNT != 12) {
        fprintf(stderr, "Qwen4Exp profile selection produced unexpected constants\n");
        model_close(&model);
        return 1;
    }

    uint32_t full = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const bool expected = (il % 4u) == 3u;
        if (ds4_qwen4exp_layer_is_full_attention(il) != expected) {
            fprintf(stderr, "Qwen4Exp layer %u has the wrong attention branch\n", il);
            model_close(&model);
            return 1;
        }
        if (expected) full++;
    }
    if (full != 12u) {
        fprintf(stderr, "Qwen4Exp full-attention layer count is %u, expected 12\n", full);
        model_close(&model);
        return 1;
    }

    ds4_str quantization = {0};
    const bool external_ple =
        model_get_string(&model, "general.quantization", &quantization) &&
        ds4_streq(quantization, "MQ-Q6-SSD-PLE-BF16");
    if (external_ple) {
        if (!config_validate_qwen4exp_external_ple(&model)) {
            fprintf(stderr, "SSD-PLE variant was not recognized as external\n");
            model_close(&model);
            return 1;
        }
        printf("Qwen3.8-Flash-Next SSD-PLE contract: valid "
               "(BF16 sidecar, 128 resident PLE tables absent, I64 controls present)\n");
    }

    ds4_weights weights;
    weights_bind(&weights, &model, false, 0, DS4_N_LAYER - 1u,
                 true, false);
    if (!weights.token_embd || !weights.output ||
        !weights.qwen_input_hc.norm ||
        !weights.qwen_mtp_fc_embedding ||
        !weights.qwen_mtp.qwen_qsa.output) {
        fprintf(stderr, "Qwen4Exp top-level or MTP weight binding is incomplete\n");
        model_close(&model);
        return 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &weights.layer[il];
        if (!l->qwen_attn_hc.inject || !l->qwen_ffn_hc.inject ||
            !l->ffn_gate_exps || !l->ffn_down_exps ||
            !l->ffn_down_exps_tail || !l->ffn_shexp_gate_inp) {
            fprintf(stderr, "Qwen4Exp layer %u weight binding is incomplete\n", il);
            model_close(&model);
            return 1;
        }
        if (ds4_qwen4exp_layer_is_full_attention(il)) {
            if (!l->qwen_qsa.output || l->qwen_linear_attn.out) {
                fprintf(stderr, "Qwen4Exp layer %u bound the wrong attention branch\n", il);
                model_close(&model);
                return 1;
            }
        } else if (!l->qwen_linear_attn.out || l->qwen_qsa.output) {
            fprintf(stderr, "Qwen4Exp layer %u bound the wrong attention branch\n", il);
            model_close(&model);
            return 1;
        }
        if ((il == 1u) != (l->qwen_ple.key != NULL)) {
            fprintf(stderr, "Qwen4Exp layer %u has the wrong PLE binding\n", il);
            model_close(&model);
            return 1;
        }
    }

    printf("Qwen3.8-Flash-Next metadata: valid "
           "(48 layers, 36 GDN + 12 QSA, 512E top-10, PLE+vision+MTP present)\n");
    printf("Qwen3.8-Flash-Next text weights: valid "
           "(48 attention/HC/MoE blocks, PLE projections+controls, MTP bound)\n");
    model_close(&model);
    return 0;
}
