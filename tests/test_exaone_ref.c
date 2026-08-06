/* K-EXAONE CPU reference forward check.
 *
 * Runs one prompt through the exaone-moe reference path and prints the router
 * decision for each MoE layer. Compared against llama.cpp's ffn_moe_topk /
 * ffn_moe_weights_norm dump for the same model and the same token, this
 * isolates "is the architecture implemented correctly" from "does the quant
 * change which experts fire" -- point both at the same GGUF and the selections
 * have to match exactly.
 *
 *   ./tests/test_exaone_ref <model.gguf> <tok0> [tok1 ...]
 *
 * Prints the router state at the FIRST token position, which is what
 * benchmarks/parse_router_dump.py extracts from the llama.cpp side.
 */
#include "../ds4.c"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <tok0> [tok1 ...]\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);

    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_EXAONE_MOE) {
        fprintf(stderr, "not an exaone-moe model: %s\n", DS4_MODEL_SHAPE_NAME);
        return 1;
    }
    fprintf(stderr, "shape: %s  layers=%u embd=%u heads=%u/%u experts=%u/%u\n",
            DS4_MODEL_SHAPE_NAME, DS4_N_LAYER, DS4_N_EMBD, DS4_N_HEAD,
            DS4_N_HEAD_KV, DS4_N_EXPERT_USED, DS4_N_EXPERT);

    ds4_weights weights;
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    fprintf(stderr, "weights bound\n");

    const int n_tok = argc - 2;
    exaone_kv_cache cache;
    exaone_kv_cache_init(&cache, (uint32_t)n_tok + 8u);

    const uint32_t n_used = DS4_N_EXPERT_USED;
    const uint32_t n_layer = DS4_N_LAYER;
    g_exaone_router_sel = xcalloc((size_t)n_layer * n_used, sizeof(uint32_t));
    g_exaone_router_w   = xcalloc((size_t)n_layer * n_used, sizeof(float));

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    uint32_t *sel0 = xcalloc((size_t)n_layer * n_used, sizeof(uint32_t));
    float    *w0   = xcalloc((size_t)n_layer * n_used, sizeof(float));

    for (int i = 0; i < n_tok; i++) {
        const int tok = atoi(argv[2 + i]);
        fprintf(stderr, "token %d/%d id=%d ...\n", i + 1, n_tok, tok);
        exaone_forward_token_cpu(logits, &model, &weights, &cache, tok, (uint32_t)i);
        if (i == 0) {
            memcpy(sel0, g_exaone_router_sel, (size_t)n_layer * n_used * sizeof(uint32_t));
            memcpy(w0,   g_exaone_router_w,   (size_t)n_layer * n_used * sizeof(float));
        }
    }

    printf("{\n  \"source\": \"ds4-reference\",\n  \"n_expert_used\": %u,\n  \"layers\": {\n",
           n_used);
    bool first = true;
    for (uint32_t il = 0; il < n_layer; il++) {
        bool any = false;
        for (uint32_t i = 0; i < n_used; i++) if (w0[il * n_used + i] != 0.0f) any = true;
        if (!any) continue;
        if (!first) printf(",\n");
        first = false;
        printf("   \"%u\": {\"selected_experts\": [", il);
        for (uint32_t i = 0; i < n_used; i++)
            printf("%s%u", i ? ", " : "", sel0[il * n_used + i]);
        printf("], \"weights\": [");
        for (uint32_t i = 0; i < n_used; i++)
            printf("%s%.6f", i ? ", " : "", w0[il * n_used + i]);
        printf("]}");
    }
    printf("\n  },\n");

    /* top-5 logits at the last position, a second comparison point */
    printf("  \"top_logits\": [");
    for (int r = 0; r < 5; r++) {
        int best = 0;
        for (int i = 1; i < (int)DS4_N_VOCAB; i++) if (logits[i] > logits[best]) best = i;
        printf("%s{\"id\": %d, \"logit\": %.4f}", r ? ", " : "", best, logits[best]);
        logits[best] = -INFINITY;
    }
    printf("]\n}\n");

    free(w0); free(sel0); free(logits);
    free(g_exaone_router_w); free(g_exaone_router_sel);
    exaone_kv_cache_free(&cache);
    model_close(&model);
    return 0;
}
