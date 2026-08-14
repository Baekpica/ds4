/* Solar Open2 full-model CUDA graph assembly regression.
 *
 *   ./tests/test_solar_forward <first-model-shard.gguf> <tok0> <tok1> [...]
 *
 * Primitive tests cover KDA, packed GQA KV, routing, and routed matmuls in
 * isolation.  This test covers the integration boundary: the common plain
 * residual/MoE workspace plus Solar's alternating GQA/KDA attention fronts.
 * It also requires the allocator and admission planner to agree byte-for-byte.
 */
#include "../ds4.c"

static int compare_logits(const float *prefill, const float *decode,
                          uint32_t n_vocab) {
    double error_sq = 0.0;
    double ref_sq = 0.0;
    double dot = 0.0;
    double prefill_sq = 0.0;
    uint32_t prefill_argmax = 0u;
    uint32_t decode_argmax = 0u;
    for (uint32_t i = 0; i < n_vocab; i++) {
        if (!isfinite(prefill[i]) || !isfinite(decode[i])) {
            fprintf(stderr, "non-finite logit at %u\n", i);
            return 1;
        }
        const double p = prefill[i];
        const double d = decode[i];
        const double e = p - d;
        error_sq += e * e;
        ref_sq += d * d;
        prefill_sq += p * p;
        dot += p * d;
        if (prefill[i] > prefill[prefill_argmax]) prefill_argmax = i;
        if (decode[i] > decode[decode_argmax]) decode_argmax = i;
    }
    const double rel_rms = ref_sq > 0.0 ? sqrt(error_sq / ref_sq) : INFINITY;
    const double cosine = prefill_sq > 0.0 && ref_sq > 0.0
        ? dot / sqrt(prefill_sq * ref_sq) : 0.0;
    fprintf(stderr,
            "Solar forward parity: rel_rms=%.6e 1-cos=%.6e "
            "argmax prefill=%u decode=%u\n",
            rel_rms, 1.0 - cosine, prefill_argmax, decode_argmax);
    /* Prefill uses tiled MMQ while single-token decode uses its vector tier.
     * Low-bit routed experts can therefore accumulate small width-dependent
     * drift even when the graph and recurrent/KV state are equivalent.  Keep
     * a magnitude guard, a direction guard, and the greedy decision pinned. */
    return rel_rms <= 4.0e-2 && 1.0 - cosine <= 1.0e-3 &&
           prefill_argmax == decode_argmax ? 0 : 1;
}

static int check_allocation_accounting(const ds4_solar_gpu_graph *g,
                                       uint32_t ctx) {
    const ds4_context_memory plan =
        ds4_context_memory_estimate(DS4_BACKEND_CUDA, (int)ctx);
    const uint64_t state = g->state_bytes + g->control_bytes;
    const uint64_t scratch = g->base.workspace_bytes + g->scratch_bytes;
    const uint64_t total = g->base.kv_bytes + state + scratch;
    if (plan.raw_bytes != g->base.kv_bytes ||
        plan.compressed_bytes != state ||
        plan.scratch_bytes != scratch ||
        plan.total_bytes != total) {
        fprintf(stderr,
                "Solar allocation/accounting mismatch: "
                "plan=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
                " actual=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
                "\n",
                plan.raw_bytes, plan.compressed_bytes, plan.scratch_bytes,
                plan.total_bytes, g->base.kv_bytes, state, scratch, total);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <first-model-shard.gguf> <tok0> <tok1> [...]\n",
                argv[0]);
        return 2;
    }
    const uint32_t n_tokens = (uint32_t)(argc - 2);
    int *tokens = xcalloc(n_tokens, sizeof(*tokens));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = atoi(argv[i + 2]);

    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        free(tokens);
        return 1;
    }
    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_SOLAR_OPEN2) {
        fprintf(stderr, "not a Solar Open2 model\n");
        model_close(&model);
        free(tokens);
        return 1;
    }
    ds4_weights weights;
    weights_bind(&weights, &model);
    if (!ds4_gpu_set_model_map(model.map, model.size)) {
        fprintf(stderr, "model map registration failed\n");
        model_close(&model);
        free(tokens);
        return 1;
    }

    const uint32_t ctx = n_tokens + 64u;
    char chunk_text[32];
    snprintf(chunk_text, sizeof(chunk_text), "%u", n_tokens);
    if (setenv("DS4_METAL_PREFILL_CHUNK", chunk_text, 1) != 0 ||
        setenv("DS4_SOLAR_KV_FORMAT", "hybrid", 1) != 0) {
        perror("setenv");
        model_close(&model);
        free(tokens);
        return 1;
    }
    float *prefill = xcalloc(DS4_N_VOCAB, sizeof(*prefill));
    float *decode = xcalloc(DS4_N_VOCAB, sizeof(*decode));
    ds4_solar_gpu_graph graph;
    int failed = 0;

    if (!solar_graph_alloc(&graph, &model, &weights, ctx, n_tokens)) {
        fprintf(stderr, "Solar graph allocation failed\n");
        failed = 1;
        goto cleanup;
    }
    failed |= check_allocation_accounting(&graph, ctx);
    if (!failed &&
        (!solar_graph_prefill_chunk(&graph, &model, &weights, tokens,
                                    n_tokens, 0u, true) ||
         !ds4_gpu_tensor_read(graph.base.logits, 0, prefill,
                              (uint64_t)DS4_N_VOCAB * sizeof(*prefill)))) {
        fprintf(stderr, "Solar full prefill failed\n");
        failed = 1;
    }
    solar_graph_free(&graph);

    if (!failed &&
        !solar_graph_alloc(&graph, &model, &weights, ctx, n_tokens)) {
        fprintf(stderr, "Solar decode graph allocation failed\n");
        failed = 1;
    }
    if (!failed &&
        (!solar_graph_prefill_chunk(&graph, &model, &weights, tokens,
                                    n_tokens - 1u, 0u, false) ||
         !solar_graph_decode(&graph, &model, &weights,
                             tokens[n_tokens - 1u], n_tokens - 1u) ||
         !ds4_gpu_tensor_read(graph.base.logits, 0, decode,
                              (uint64_t)DS4_N_VOCAB * sizeof(*decode)))) {
        fprintf(stderr, "Solar prefix+decode failed\n");
        failed = 1;
    }
    if (!failed) failed = compare_logits(prefill, decode, DS4_N_VOCAB);
    solar_graph_free(&graph);

cleanup:
    free(decode);
    free(prefill);
    model_close(&model);
    free(tokens);
    ds4_gpu_cleanup();
    puts(failed ? "Solar forward integration FAILED"
                : "Solar forward integration passed");
    return failed ? 1 : 0;
}
