/* K-EXAONE full forward: CUDA graph against the CPU reference.
 *
 *   ./tests/test_exaone_forward <model.gguf> <tok0> [tok1 ...]
 *
 * The kernel checks next to this file test each stage in isolation. This one
 * tests the assembly: embedding, 48 layers of attention and MoE, the final
 * norm and the output head, on the real artifact, comparing the CUDA graph
 * against the CPU reference forward in ds4.c token for token.
 *
 * Both sides read the same weights, so this isolates the engine from the
 * quantization: a difference here is a kernel or a graph-wiring bug, not the
 * recipe. Tokens are given as ids so the tokenizer is not in the loop.
 *
 * Prefill and decode are checked separately against the same reference,
 * because they take different paths through exaone_graph_layer (batched
 * attention and the tiled matmul tier, versus the single-row entries) and a
 * bug in one would otherwise hide behind the other.
 */
#include "../ds4.c"

static int g_fail = 0;

/* Logit agreement. mmq quantizes activations to Q8_1 in the routed-expert
 * matmuls while the CPU reference keeps f32 throughout, so the two cannot
 * agree bit for bit; what has to hold is that the noise stays at the
 * quantization floor and the argmax is unchanged. */
static void compare_logits(const char *what, const float *gpu, const float *cpu,
                           int n, double tol_rel_rms) {
    double se = 0.0, sr = 0.0, sg = 0.0, dot = 0.0;
    int amax_g = 0, amax_c = 0;
    for (int i = 0; i < n; i++) {
        const double g = gpu[i], c = cpu[i], d = g - c;
        se += d * d; sr += c * c; sg += g * g; dot += g * c;
        if (gpu[i] > gpu[amax_g]) amax_g = i;
        if (cpu[i] > cpu[amax_c]) amax_c = i;
    }
    const double rel_rms = sr > 0.0 ? sqrt(se / sr) : INFINITY;
    const double cosv = (sg > 0.0 && sr > 0.0) ? dot / sqrt(sg * sr) : 0.0;
    const int ok = rel_rms <= tol_rel_rms && amax_g == amax_c;
    if (!ok) g_fail++;
    printf("%-26s rel_rms=%.3e 1-cos=%.3e argmax gpu=%d cpu=%d  %s\n",
           what, rel_rms, 1.0 - cosv, amax_g, amax_c, ok ? "ok" : "FAIL");
}

static void top5(const char *tag, const float *l, int n) {
    printf("  %-8s", tag);
    float *tmp = xmalloc((size_t)n * sizeof(float));
    memcpy(tmp, l, (size_t)n * sizeof(float));
    for (int r = 0; r < 5; r++) {
        int b = 0;
        for (int i = 1; i < n; i++) if (tmp[i] > tmp[b]) b = i;
        printf(" %d:%.3f", b, tmp[b]);
        tmp[b] = -INFINITY;
    }
    printf("\n");
    free(tmp);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <tok0> [tok1 ...]\n", argv[0]);
        return 2;
    }
    const int n_tok = argc - 2;
    int *toks = xcalloc((size_t)n_tok, sizeof(int));
    for (int i = 0; i < n_tok; i++) toks[i] = atoi(argv[2 + i]);

    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_EXAONE_MOE) {
        fprintf(stderr, "not an exaone-moe model: %s\n", DS4_MODEL_SHAPE_NAME);
        return 1;
    }
    ds4_weights weights;
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    if (!ds4_gpu_set_model_map(model.map, model.size)) {
        fprintf(stderr, "model map registration failed\n");
        return 1;
    }
    fprintf(stderr, "shape: %s  layers=%u embd=%u heads=%u/%u experts=%u/%u\n",
            DS4_MODEL_SHAPE_NAME, DS4_N_LAYER, DS4_N_EMBD, DS4_N_HEAD,
            DS4_N_HEAD_KV, DS4_N_EXPERT_USED, DS4_N_EXPERT);

    const int n_vocab = (int)DS4_N_VOCAB;
    float *cpu_logits = xmalloc((size_t)n_vocab * sizeof(float));
    float *gpu_logits = xmalloc((size_t)n_vocab * sizeof(float));

    /* ---- CPU reference: sequential decode over every token ---- */
    fprintf(stderr, "cpu reference: %d token(s)...\n", n_tok);
    {
        exaone_kv_cache cache;
        exaone_kv_cache_init(&cache, (uint32_t)n_tok + 8u);
        const double t0 = now_sec();
        for (int i = 0; i < n_tok; i++) {
            exaone_forward_token_cpu(cpu_logits, &model, &weights, &cache,
                                     toks[i], (uint32_t)i);
            fprintf(stderr, "  cpu token %d/%d (%.1fs)\n", i + 1, n_tok,
                    now_sec() - t0);
        }
        exaone_kv_cache_free(&cache);
    }

    ds4_exaone_gpu_graph g;
    const uint32_t ctx = (uint32_t)n_tok + 64u;

    /* ---- GPU prefill: the whole prompt as one chunk ---- */
    if (!exaone_graph_alloc(&g, &model, &weights, ctx, (uint32_t)n_tok)) {
        fprintf(stderr, "exaone_graph_alloc failed\n");
        return 1;
    }
    {
        const double t0 = now_sec();
        if (!exaone_graph_prefill_chunk(&g, &model, &weights, toks,
                                        (uint32_t)n_tok, 0, true)) {
            fprintf(stderr, "prefill failed\n");
            return 1;
        }
        if (!ds4_gpu_tensor_read(g.logits, 0, gpu_logits,
                                 (uint64_t)n_vocab * sizeof(float))) {
            fprintf(stderr, "logits readback failed\n");
            return 1;
        }
        fprintf(stderr, "gpu prefill: %.3fs\n", now_sec() - t0);
    }
    compare_logits("prefill vs cpu", gpu_logits, cpu_logits, n_vocab, 3e-2);
    top5("gpu", gpu_logits, n_vocab);
    top5("cpu", cpu_logits, n_vocab);
    exaone_graph_free(&g);

    /* ---- GPU decode: prefill the prefix, then decode the last token ----
     * Same expected logits, reached through the single-row entries. */
    if (n_tok >= 2) {
        if (!exaone_graph_alloc(&g, &model, &weights, ctx, (uint32_t)n_tok)) {
            fprintf(stderr, "exaone_graph_alloc failed\n");
            return 1;
        }
        const double t0 = now_sec();
        if (!exaone_graph_prefill_chunk(&g, &model, &weights, toks,
                                        (uint32_t)(n_tok - 1), 0, false)) {
            fprintf(stderr, "prefix prefill failed\n");
            return 1;
        }
        const double t1 = now_sec();
        if (!exaone_graph_decode(&g, &model, &weights, toks[n_tok - 1],
                                 (uint32_t)(n_tok - 1))) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
        if (!ds4_gpu_tensor_read(g.logits, 0, gpu_logits,
                                 (uint64_t)n_vocab * sizeof(float))) {
            fprintf(stderr, "logits readback failed\n");
            return 1;
        }
        const double t2 = now_sec();
        fprintf(stderr, "gpu prefix prefill: %.3fs, decode step: %.3fs\n",
                t1 - t0, t2 - t1);
        compare_logits("decode vs cpu", gpu_logits, cpu_logits, n_vocab, 3e-2);
        top5("gpu", gpu_logits, n_vocab);
        exaone_graph_free(&g);
    }

    free(gpu_logits); free(cpu_logits); free(toks);
    model_close(&model);
    printf("\n%s\n", g_fail ? "FAILURES" : "all checks passed");
    return g_fail ? 1 : 0;
}
